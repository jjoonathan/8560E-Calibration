import pickle
import holoviews as hv
import scipy.signal
import numpy as np
import pandas as pd
from HP8560Eeprom import HP8560Eeprom

class HP8560SweepResults:

    #################################################### Save & Load ###############################################################
    @classmethod
    def read_pickle(cls, pickle_file_name):
        with open(pickle_file_name,'rb') as f:
            loadswp = pickle.load(f)
        full_span_sweeps = loadswp['full_span_sweeps']
        narrow_sweeps = loadswp['narrow_sweeps']
        dac_sweeps = loadswp['dac_sweeps']
        return cls(full_span_sweeps, narrow_sweeps, dac_sweeps)
    
    def __init__(self, full_span_sweeps, narrow_sweeps, dac_sweeps):
        self.full_span_sweeps = full_span_sweeps
        self.narrow_sweeps = narrow_sweeps
        self.dac_sweeps = dac_sweeps
        self.dac_maps = None

    def to_pickle(self, fname):
        save_dict = dict(
            full_span_sweeps=self.full_span_sweeps,
            narrow_sweeps=self.narrow_sweeps,
            dac_sweeps=self.dac_sweeps
        )
        with open(fname,'wb') as f:
            pickle.dump(save_dict, f)


    #################################################### YTF Optimization ###############################################################
    def optim_path(self, dac_dBm, smoothing, learning_rate, num_iterations, initial_offset, lr, betas, history=None, seed=None):
        import torch
        import torch.optim as optim
        if seed is not None:
            torch.manual_seed(seed)
        
        num_points = dac_dBm.shape[1]
        x_coords = torch.arange(num_points)
        dac_dBm = torch.tensor(dac_dBm)
    
        def bilinear_interpolation(x, y, field):
            x0 = torch.floor(x).long()
            x1 = x0 + 1
            y0 = torch.floor(y).long()
            y1 = y0 + 1
        
            x0 = x0.clamp(0, field.shape[1] - 1)
            x1 = x1.clamp(0, field.shape[1] - 1)
            y0 = y0.clamp(0, field.shape[0] - 1)
            y1 = y1.clamp(0, field.shape[0] - 1)
        
            Ia = field[y0, x0]
            Ib = field[y1, x0]
            Ic = field[y0, x1]
            Id = field[y1, x1]
        
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)
        
            return wa * Ia + wb * Ib + wc * Ic + wd * Id
        
        # Path(x)
        p_x = initial_offset + torch.randn(dac_dBm.shape[1])*20
        p_x.requires_grad = True
        
        optimizer = optim.Adam([p_x], lr=lr, betas=betas)
        #optimizer = optim.SGD([p_x], lr=1)
        for i in range(num_iterations):
            optimizer.zero_grad()
            path_values = bilinear_interpolation(x_coords, p_x, dac_dBm)
            L_dBm  = -path_values.mean()
            L_bump = p_x.diff().pow(2).mean()*smoothing
            L = L_dBm + L_bump
            L.backward()
            optimizer.step()
    
            if i % 10 == 0:
                #print(f"Iteration {i}, L_dBm={L_dBm.item()}, L_bump={L_bump.item()}")
                if history is not None:
                    history.append(np.copy(p_x.detach().numpy()))
        optimized_path = np.round(p_x.detach().numpy()).astype(int)
        return optimized_path
    
    def optim_band(self, band_name, history=None, seed=None, initial_offset_=None):
        # history: if you pass an array, snapshots of p_x will be added every 10 step()s.
        self.populate_dac_maps()
        dac_dBm = scipy.ndimage.convolve1d(self.dac_maps[band_name], np.ones(1), axis=0)
        smoothing = 1.0
        learning_rate = 1.0
        num_iterations = 1000
        initial_offset = 128
        lr = 1
        betas = (0.9,0.99)
        if band_name=='band3':
            initial_offset = 100 # 100 good 128 bad ---- Added for device 248 to prevent peak hopping
        if band_name=='band4':
            smoothing = 0.015
            initial_offset = 60
        if band_name=='band5':
            smoothing = 0.015
            initial_offset = 60
        if initial_offset_ is not None:
            initial_offset = initial_offset_
        p_x = self.optim_path(dac_dBm,
                         smoothing=smoothing,
                         learning_rate=learning_rate, 
                         num_iterations=num_iterations,
                         history=history,
                         initial_offset=initial_offset,
                         lr=lr,
                         betas=betas,
                         seed=seed
                        )
        return p_x

    def optim_bands(self, seed=42):
        self.populate_dac_maps()
        self.band_cals = {band_name:self.optim_band(band_name,seed=seed) for band_name in self.swp_counter.keys()}

    def update_eeprom_ytf(self, eep : HP8560Eeprom):
        self.band_cals # Populate by calling self.optim_bands()
        for band_name, dacs in self.band_cals.items():
            for i in range(len(dacs)):
                eep.set_ytf_correction_pt(band_name, i, dacs[i])
        eep.update_correction()
        return eep.plt_ytf()

    def plt_optim_convergence(self, band_name, discrete=False, seed=42, initial_offset_=None):
        # Perform optimization, capture trace
        history = []
        p_opt = self.optim_band(band_name,history=history,seed=seed,initial_offset_=initial_offset_)
        # Interactive Display of Convergence Process
        ghz_min, ghz_max = self.swp_range[band_name]
        dac_dBm = self.dac_maps[band_name]
        h,w = dac_dBm.shape
        img = hv.Image(dac_dBm[::-1,:],bounds=(ghz_min,0,ghz_max,h),kdims=['GHz','DAC'])
        def plt_path_history(i):
            x = self.swp_freqs[band_name]
            crv = hv.Curve((x,history[i]))
            return img*crv.opts(width=800)
        dmap = hv.DynamicMap(plt_path_history,kdims=[hv.Dimension('optimization_step//10',range=(0,len(history)-1))]).opts(title='Convergence Monitor')
        hmap = hv.HoloMap(dmap[list(range(0,len(history),10))])
        return hmap if discrete else dmap

    # Interactive Display of Final DAC Settings
    def plt_optim(self, band_name, discrete=False, seed=42, initial_offset_=None):
        p_opt = self.optim_band(band_name,seed=seed,initial_offset_=initial_offset_)
        dac_dBm = self.dac_maps[band_name]
        ghz_min, ghz_max = self.swp_range[band_name]
        h,w = dac_dBm.shape
        def plt_xsection(i):
            h,w = dac_dBm.shape
            img = hv.Image(dac_dBm[::-1,:],bounds=(ghz_min,0,ghz_max,h),kdims=['GHz','DAC']).opts(width=1000,tools=['hover'])
            x = self.swp_freqs[band_name]
            vline = hv.VLine(x[i]).opts(line_width=2,line_color='r')
            crv_soln = hv.Curve((x,p_opt))
            dac = int(p_opt[i])
            crv_slice = hv.Curve((np.arange(h),dac_dBm[:,i]),kdims=['DAC'],vdims=['dBm']).opts(width=1000,tools=['hover'])
            pt  = hv.Points([(dac,dac_dBm[dac,i])]).opts(size=5,color='r')
            return (img*vline*crv_soln+crv_slice*pt).cols(1)
        dmap = hv.DynamicMap(plt_xsection,kdims=[hv.Dimension('i',range=(0,dac_dBm.shape[1]-1))])
        hmap = hv.HoloMap(dmap[list(range(0,dac_dBm.shape[1],10))]).collate()
        return hmap if discrete else dmap

    def plt_optim_bands(self):
        self.band_cals # Populate by calling self.optim_bands()
        def disp_band_cal(band_name):
            dacs = self.band_cals[band_name]
            ghzs = self.swp_freqs[band_name]
            ghz_min, ghz_max = self.swp_range[band_name]
            dac_dBm = self.dac_maps[band_name]
            h,w = dac_dBm.shape
            img = hv.Image(dac_dBm[::-1,:],bounds=(ghz_min,0,ghz_max,h),kdims=[f'GHz ({band_name})','DAC'])
            crv = hv.Curve((ghzs,dacs),kdims=[f'GHz ({band_name})']).opts(tools=['hover'])
            return (img*crv).opts(width=800)
        return hv.Layout([disp_band_cal(band_name) for band_name in self.band_cals.keys()]).cols(1)

    #################################################### Plots ###############################################################
    def plt_in_sweeps(self):
        full_span_sweeps = self.full_span_sweeps
        narrow_sweeps = self.narrow_sweeps
        dac_sweeps = self.dac_sweeps
        img_2d = hv.Image(np.array([ns['trace_dB'] for ns in narrow_sweeps])[::-1,:], kdims=['MHz','Sweep#'], bounds=(-5,0,5,len(narrow_sweeps))).opts(tools=['hover'])
        freqs_ghz = [ns['f_hz']/1e9 for ns in narrow_sweeps]
        def plt_swp_f(i):
            f_GHz = freqs_ghz[i]
            meas = narrow_sweeps[i]
            fs = meas['trace_f']
            map_2d = img_2d*hv.HLine(i).opts(line_width=1,color='red')
            return map_2d+hv.Curve((fs-fs[300],meas['trace_dB'])).opts(width=800,show_grid=True,yticks=10,title=f'{f_GHz}')
        return hv.DynamicMap(plt_swp_f, kdims=[hv.Dimension('Sweep#',range=(0,len(narrow_sweeps)-1))])
    
    def plt_out_sweeps(self):
        full_span_sweeps = self.full_span_sweeps
        narrow_sweeps = self.narrow_sweeps
        dac_sweeps = self.dac_sweeps
        img_2d = hv.Image(np.array([fss['trace_dB'] for fss in full_span_sweeps])[::-1,:], kdims=['MHz','Sweep#'], bounds=(-5,0,5,len(narrow_sweeps))).opts(tools=['hover'])
        freqs_ghz = [fss['f_hz']/1e9 for fss in full_span_sweeps]
        def plt_swp_f(i):
            f_GHz = freqs_ghz[i]
            meas = full_span_sweeps[i]
            fs = meas['trace_f']
            map_2d = img_2d*hv.HLine(i).opts(line_width=1,color='red')
            return map_2d+hv.Curve((fs-fs[300],meas['trace_dB'])).opts(width=800,show_grid=True,yticks=10,title=f'{f_GHz}')
        return hv.DynamicMap(plt_swp_f, kdims=[hv.Dimension('Sweep#',range=(0,len(narrow_sweeps)-1))])
    
    def plt_dac_sweeps(self):
        dac_sweeps = self.dac_sweeps
        def plt_dac_sweep(idx):
            dsi = dac_sweeps[idx]
            x = dsi['dac_sweep_dac']
            y = dsi['dac_sweep_dBm']
            GHz = dsi['f_hz']/1e9
            return hv.Curve((x,y)).opts(title=f'{GHz} GHz ({dsi["band"]}-{dsi["band_i"]})').opts(xlim=(0,255))
        
        return hv.DynamicMap(plt_dac_sweep, kdims=[hv.Dimension('idx',range=(0,len(dac_sweeps)-1))])

    def populate_dac_maps(self, discard_border_px_=5):
        if self.dac_maps is not None:
            return
        dac_sweeps = self.dac_sweeps
    
        # swp_freqs['band1'] = [1.23, 1.24, ...]  where 1.23 is frequency in GHz
        swp_freqs = {}
        for ds in dac_sweeps:
            band_name = ds['band']
            arr = swp_freqs.get(band_name,[])
            arr.append(ds['f_hz']/1e9)
            swp_freqs[band_name] = arr
        for sf in swp_freqs.values():
            sf[:] = sorted(sf[:])
        
        swp_range = {band_name:(np.min(swp_freqs[band_name]),np.max(swp_freqs[band_name])) for band_name in swp_freqs.keys()}
        
        # swp_counter['band1'] = # sweeps in band1
        swp_counter = {band_name:len(band_freqs) for band_name,band_freqs in swp_freqs.items()}
        if 'band0' in swp_counter:
            del swp_counter['band0']
        
        # dac_maps['band1'] -> 256 x swp_counter['band1']
        dac_maps = {}
        for band_name,band_count in swp_counter.items():
            dac_dBm = np.full((256,band_count),np.nan)
            dac_maps[band_name] = dac_dBm
        
        # Fill observed DAC values
        for ds in dac_sweeps:
            if ds['band']=='band0':
                continue
            dac_dBm = dac_maps[ds['band']]
            dac = ds['dac_sweep_dac']
            dBm = ds['dac_sweep_dBm']
            band_i = ds['band_i']
            dac_dBm[dac,band_i] = dBm
            dBm_min = np.min(dBm)
            dac_dBm[0:(np.min(dac)+discard_border_px_-1),band_i] = dBm_min
            dac_dBm[(np.max(dac)-discard_border_px_+1):256,band_i] = dBm_min
        
        # Fill unobserved DAC values
        dac_maps_2 = {}
        for band_name,dac_dBm in dac_maps.items():
            df = pd.DataFrame(dac_dBm)
            df.interpolate(method='linear',direction='nearest',limit_direction='both',axis=0,inplace=True)
            dac_maps_2[band_name] = df.to_numpy()
        dac_maps = dac_maps_2
        self.swp_freqs = swp_freqs
        self.swp_range = swp_range
        self.swp_counter = swp_counter
        self.dac_maps = dac_maps

    def plt_dac_sweeps_2d(self):
        self.populate_dac_maps()
        def plt_item(band_name,dac_dBm):
            ghz_min, ghz_max = self.swp_range[band_name]
            xdim = hv.Dimension(f'Frequency ({band_name}, GHz)', range=(ghz_min,ghz_max))
            ydim = hv.Dimension('DAC')
            h,w = dac_dBm.shape
            return hv.Image(dac_dBm[::-1,:],kdims=[xdim,ydim],bounds=(ghz_min,0,ghz_max,h)).opts(width=800,tools=['hover'])
        return hv.Layout([plt_item(band_name,dac_dBm) for band_name,dac_dBm in self.dac_maps.items()]).cols(1)
    
    def plt_sweeps(self, show_i=True, show_o=True, show_p=True, smooth_n=1, title=''):
        full_span_sweeps = self.full_span_sweeps
        narrow_sweeps = self.narrow_sweeps
        dac_sweeps = self.dac_sweeps
        band_ytfs = {b:([],[]) for b in set([ns['band'] for ns in narrow_sweeps])}
        for ds in dac_sweeps:
            band_ytfs[ds['band']][0].append(ds['f_hz']/1e9)
            band_ytfs[ds['band']][1].append(int(ds['psdac']))
        band_ytfs_smoothed = {}
        for band,(ghz,ytfs) in band_ytfs.items():
            if len(ytfs)<=1:
                continue
            if smooth_n<=1:
                ytfs_smoothed = ytfs
            if smooth_n>=2:
                ytfs_smoothed = scipy.signal.filtfilt(np.ones(smooth_n)/smooth_n, [1], ytfs)
            band_ytfs_smoothed[band] = (ghz, ytfs_smoothed)
        band_ytfs = band_ytfs_smoothed
        
        ytf_curves = [
            hv.Curve((x,y), kdims=['GHz'], vdims=['ytf'], label=band).opts(width=1000, tools=['hover'], show_grid=True)
            for band,(x,y) in band_ytfs.items()
        ]

        if show_i:
            band_dbms_i = {b:([],[]) for b in set([ns['band'] for ns in narrow_sweeps])} # zoomed In
            for ns in narrow_sweeps:
                band_dbms_i[ns['band']][0].append(ns['f_hz']/1e9)
                band_dbms_i[ns['band']][1].append(np.max(ns['trace_dB'])-ns['power_meter_dBm'])
            dbm_i_curves = [
                hv.Curve((x,y), kdims=['GHz'], vdims=['dB'], label='I'+band[-1]).opts(width=1000, tools=['hover'], show_grid=True)
                for band,(x,y) in band_dbms_i.items()
            ]
        if not show_i:
            dbm_i_curves = []

        if show_o:
            band_dbms_o = {b:([],[]) for b in set([ns['band'] for ns in narrow_sweeps])} # zoomed Out
            for fs in full_span_sweeps:
                band_dbms_o[fs['band']][0].append(fs['f_hz']/1e9)
                band_dbms_o[fs['band']][1].append(np.max(fs['trace_dB'][1:])-fs['power_meter_dBm'])
            dbm_o_curves = [
                hv.Curve((x,y), kdims=['GHz'], vdims=['dB'], label='O'+band[-1]).opts(width=1000, tools=['hover'], show_grid=True)
                for band,(x,y) in band_dbms_o.items()
            ]
        if not show_o:
            dbm_o_curves = []

        if show_p:
            band_dbms_p = {b:([],[]) for b in set([ns['band'] for ns in narrow_sweeps])} # Preselected
            for ds in dac_sweeps:
                band_dbms_p[ds['band']][0].append(ds['f_hz']/1e9)
                band_dbms_p[ds['band']][1].append(np.max(ds['trace_dB'])-ds['power_meter_dBm'])
            dbm_p_curves = [
                hv.Curve((x,y), kdims=['GHz'], vdims=['dB'], label='P'+band[-1]).opts(width=1000, tools=['hover'], show_grid=True)
                for band,(x,y) in band_dbms_p.items()
            ]
        if not show_p:
            dbm_p_curves = []
        
        return hv.Layout((
            hv.Overlay(ytf_curves).opts(legend_position='top',title=title),
            hv.Overlay(dbm_p_curves+dbm_i_curves+dbm_o_curves).opts(legend_position='top',title=title)
        )).cols(1)