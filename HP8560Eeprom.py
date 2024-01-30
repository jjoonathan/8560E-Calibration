import struct, pickle
import holoviews as hv
import pandas as pd, numpy as np
import tqdm
import scipy.signal

class HP8560Eeprom:
    # Chip: X2864AP
    @classmethod
    def from_file(cls, path):
        with open(path,'rb') as f:
            eep = f.read()
        return cls(eep)

    def to_file(self, path):
        with open(path,'wb') as f:
            f.write(self.eep)

    @classmethod
    def from_gpib(cls, vxi11_Instrument):
        spec_an = vxi11_Instrument
        spec_an.clear()
        spec_an.write(f'ZSETADDR {0xFF0000}')
        barr = bytearray()
        for i in tqdm.tqdm(range(0x2000)): # 0x2000
            barr.append(int(spec_an.ask('ZRDWR?')))
        eep = bytes(barr)
        return cls(eep)

    def to_gpib(self, vxi11_Instrument):
        spec_an = vxi11_Instrument
        spec_an.write(f'ZSETADDR {0xFF0000}')
        for i in tqdm.tqdm(range(0x2000)): # 0x2000
            spec_an.write(f'ZRDWR {self.eep[i]}')
            if i%10==0:
                spec_an.ask('ZRFCAL?')
    
    def __init__(self, eep):
        self.eep = eep
        self.sn = str(eep[0x1000:0x100A],'utf8')
        self.model = str(eep[0x100C:0x1049].split(b'\x00')[0],'utf8')
        
        band0 = pd.DataFrame(np.frombuffer(eep[0x108A:0x114D], np.dtype([('cdB','>H'),('pad','B')])))
        band0['MHz'] = np.concatenate((np.array([2,6,10,27,44,61,78,95]), np.arange(100,2950,50)))
        band0['cdB_new'] = band0['cdB']
        self.band0 = band0
        
        band1 = pd.DataFrame(np.frombuffer(eep[0x114D:0x1228], np.dtype([('cdB','>H'),('ytf','B')])))
        band1['MHz'] = np.arange(2800,6450,50)
        band1['cdB_new'] = band1['cdB']
        band1['ytf_new'] = band1['ytf']
        self.band1 = band1
        
        band2 = pd.DataFrame(np.frombuffer(eep[0x1228:0x13ED], np.dtype([('cdB','>H'),('ytf','B')])))
        band2['MHz'] = np.arange(5700,13250,50)
        band2['cdB_new'] = band2['cdB']
        band2['ytf_new'] = band2['ytf']
        self.band2 = band2
        
        band3 = pd.DataFrame(np.frombuffer(eep[0x13ED:0x1756], np.dtype([('cdB','>H'),('ytf','B')])))
        band3['MHz'] = np.arange(12400,26950,50)
        band3['cdB_new'] = band3['cdB']
        band3['ytf_new'] = band3['ytf']
        self.band3 = band3
        
        band4 = pd.DataFrame(np.frombuffer(eep[0x1756:0x179E], np.dtype([('cdB','>H'),('ytf','B')])))
        band4['MHz'] = np.arange(26450,31250,200)
        band4['cdB_new'] = band4['cdB']
        band4['ytf_new'] = band4['ytf']
        self.band4 = band4
        
        band5 = pd.DataFrame(np.frombuffer(eep[0x179E:0x18BB], np.dtype([('cdB','>H'),('ytf','B')])))
        band5['MHz'] = np.arange(31100,50100,200)
        band5['cdB_new'] = band5['cdB']
        band5['ytf_new'] = band5['ytf']
        self.any_new = False
        self.band5 = band5
        self.bands = dict(band0=band0, band1=band1, band2=band2, band3=band3, band4=band4, band5=band5)

    def set_dB_correction_pt(self, band_name, band_i, correction_dB):
        self.any_new = True
        df = self.bands[band_name]
        cdB_old = df['cdB'].iloc[band_i]
        df.loc[band_i, 'cdB_new'] = cdB_old + int(correction_dB*100)

    def set_ytf_correction_pt(self, band_name, band_i, ytf_new):
        self.any_new = True
        df = self.bands[band_name]
        if ytf_new is not None and band_name != 'band0':
            df.loc[band_i, 'ytf_new'] = ytf_new

    def update_correction(self):
        new_eep = bytearray(self.eep)
        assert(len(new_eep)==0x2000)
        df = self.band0
        vals = list(zip(df['cdB_new'].values, np.zeros_like(df['cdB_new'].values)))
        buf = np.array(vals, dtype=[('cdB','>H'),('ytf','B')]).tobytes()
        assert(len(buf)==195)
        new_eep[0x108A:0x114D] = buf
        
        df = self.band1
        vals = list(zip(df['cdB_new'].values, df['ytf_new'].values))
        buf = np.array(vals, dtype=[('cdB','>H'),('ytf','B')]).tobytes()
        assert(len(buf)==219)
        new_eep[0x114D:0x1228] = buf
        
        df = self.band2
        vals = list(zip(df['cdB_new'].values, df['ytf_new'].values))
        buf = np.array(vals, dtype=[('cdB','>H'),('ytf','B')]).tobytes()
        assert(len(buf)==453)
        new_eep[0x1228:0x13ED] = buf
        
        df = self.band3
        vals = list(zip(df['cdB_new'].values, df['ytf_new'].values))
        buf = np.array(vals, dtype=[('cdB','>H'),('ytf','B')]).tobytes()
        assert(len(buf)==873)
        new_eep[0x13ED:0x1756] = buf
        
        df = self.band4
        vals = list(zip(df['cdB_new'].values, df['ytf_new'].values))
        buf = np.array(vals, dtype=[('cdB','>H'),('ytf','B')]).tobytes()
        assert(len(buf)==72)
        new_eep[0x1756:0x179E] = buf
        
        df = self.band5
        vals = list(zip(df['cdB_new'].values, df['ytf_new'].values))
        buf = np.array(vals, dtype=[('cdB','>H'),('ytf','B')]).tobytes()
        assert(len(buf)==285)
        new_eep[0x179E:0x18BB] = buf

        assert(len(new_eep)==0x2000)
        self.eep = bytes(new_eep)
        self.update_csum()

    def smooth_correction(self, smooth_n):
        if smooth_n<=1:
            return
        for band_df in [self.band1, self.band2, self.band3, self.band4, self.band5]:
            ytfs = band_df['ytf_new'].values
            ytfs1 = scipy.signal.filtfilt(np.ones(smooth_n)/smooth_n, [1], ytfs)
            band_df['ytf_new'] = ytfs1

    def update_from_swp(self, swp, smooth_n=1, update_ytf=True, unpeaked_amplitude=False):
        # unpeaked_amplitude: the correction should use the amplitudes measured before peaking YTF
        # full_span_sweeps: [{'f_hz', 'power_meter_dBm', 'trace_dB', 'trace_f', 'band', 'band_i'},...]
        # narrow_sweeps:    [{'f_hz', 'power_meter_dBm', 'trace_dB', 'trace_f', 'band', 'band_i'},...]
        # dac_sweeps:       [{'f_hz', 'power_meter_dBm', 'trace_dB', 'trace_f', 'psdac', 'band', 'band_i'},...]
        for ns in swp['narrow_sweeps']: # Narrow Sweeps. Made with default YTF setting.
            sa_dBm = np.max(ns['trace_dB'])
            pm_dBm = ns['power_meter_dBm']
            self.set_dB_correction_pt(ns['band'], ns['band_i'], -(pm_dBm-sa_dBm))
        for ds in swp['dac_sweeps']: # DAC-Tuned Sweeps. Made after tuning preselector.
            if not unpeaked_amplitude:
                sa_dBm = np.max(ds['trace_dB'])
                pm_dBm = ds['power_meter_dBm']
                self.set_dB_correction_pt(ds['band'], ds['band_i'], -(pm_dBm-sa_dBm))
            if update_ytf:
                ytf = int(ds['psdac'])
                self.set_ytf_correction_pt(ds['band'], ds['band_i'], ytf)
        self.smooth_correction(smooth_n)
        self.update_correction()

    def update_from_pickle(self, pfile, smooth_n=1, update_ytf=True, unpeaked_amplitude=False):
        swp = pickle.load(open(pfile,'rb'))
        self.update_from_swp(swp, smooth_n=smooth_n, update_ytf=update_ytf, unpeaked_amplitude=unpeaked_amplitude)

    def compute_csum(self):
        eep = self.eep
        csum_value = int(np.sum(np.frombuffer(eep[4234:6349],dtype=np.uint8))+3)&0xFFFF
        csum_bytes = struct.pack('>H',csum_value)
        return csum_bytes

    def read_csum(self):
        return self.eep[0x18CE:0x18D0]

    def update_csum(self):
        ba = bytearray(self.eep)
        ba[0x18CE:0x18D0] = self.compute_csum()
        self.eep = bytes(ba)

    def plt_flatness(self):
        b0dB = hv.Curve((self.band0['MHz']/1000, self.band0['cdB']/-100),kdims=['GHz'],vdims=['dB'],label='band0').opts(color='gray')
        b1dB = hv.Curve((self.band1['MHz']/1000, self.band1['cdB']/-100),kdims=['GHz'],vdims=['dB'],label='band1').opts(color='gray')
        b2dB = hv.Curve((self.band2['MHz']/1000, self.band2['cdB']/-100),kdims=['GHz'],vdims=['dB'],label='band2').opts(color='gray')
        b3dB = hv.Curve((self.band3['MHz']/1000, self.band3['cdB']/-100),kdims=['GHz'],vdims=['dB'],label='band3').opts(color='gray')
        b4dB = hv.Curve((self.band4['MHz']/1000, self.band4['cdB']/-100),kdims=['GHz'],vdims=['dB'],label='band4').opts(color='gray')
        b5dB = hv.Curve((self.band5['MHz']/1000, self.band5['cdB']/-100),kdims=['GHz'],vdims=['dB'],label='band5').opts(color='gray')
        fc_old = (b0dB*b1dB*b2dB*b3dB*b4dB*b5dB).opts(hv.opts.Curve(title='Flatness Cal',line_width=1,tools=['hover'],width=1000,show_grid=True,xticks=10))
        if not self.any_new:
            return fc_old
        b0dBn = hv.Curve((self.band0['MHz']/1000, self.band0['cdB_new']/-100),kdims=['GHz'],vdims=['dB'],label='new0').opts(color='red')
        b1dBn = hv.Curve((self.band1['MHz']/1000, self.band1['cdB_new']/-100),kdims=['GHz'],vdims=['dB'],label='new1').opts(color='red')
        b2dBn = hv.Curve((self.band2['MHz']/1000, self.band2['cdB_new']/-100),kdims=['GHz'],vdims=['dB'],label='new2').opts(color='red')
        b3dBn = hv.Curve((self.band3['MHz']/1000, self.band3['cdB_new']/-100),kdims=['GHz'],vdims=['dB'],label='new3').opts(color='red')
        b4dBn = hv.Curve((self.band4['MHz']/1000, self.band4['cdB_new']/-100),kdims=['GHz'],vdims=['dB'],label='new4').opts(color='red')
        b5dBn = hv.Curve((self.band5['MHz']/1000, self.band5['cdB_new']/-100),kdims=['GHz'],vdims=['dB'],label='new5').opts(color='red')
        fc_new = (b0dBn*b1dBn*b2dBn*b3dBn*b4dBn*b5dBn).opts(hv.opts.Curve(title='Flatness Cal',line_width=1,tools=['hover'],width=1000,show_grid=True,xticks=10))
        return (fc_old*fc_new).opts(legend_position='top')

    def plt_ytf(self):
        b1ytf = hv.Curve((self.band1['MHz']/1000,self.band1['ytf']),kdims=['GHz'],vdims=['ytf'],label='band1').opts(color='gray')
        b2ytf = hv.Curve((self.band2['MHz']/1000,self.band2['ytf']),kdims=['GHz'],vdims=['ytf'],label='band2').opts(color='gray')
        b3ytf = hv.Curve((self.band3['MHz']/1000,self.band3['ytf']),kdims=['GHz'],vdims=['ytf'],label='band3').opts(color='gray')
        b4ytf = hv.Curve((self.band4['MHz']/1000,self.band4['ytf']),kdims=['GHz'],vdims=['ytf'],label='band4').opts(color='gray')
        b5ytf = hv.Curve((self.band5['MHz']/1000,self.band5['ytf']),kdims=['GHz'],vdims=['ytf'],label='band5').opts(color='gray')
        yc_old = (b1ytf*b2ytf*b3ytf*b4ytf*b5ytf).opts(hv.opts.Curve(line_width=1,title='YTF Cal',tools=['hover'],width=1000,show_grid=True,xticks=10))
        if not self.any_new:
            return yc_old
        b1ytfN = hv.Curve((self.band1['MHz']/1000,self.band1['ytf_new']),kdims=['GHz'],vdims=['ytf'],label='new1').opts(color='red')
        b2ytfN = hv.Curve((self.band2['MHz']/1000,self.band2['ytf_new']),kdims=['GHz'],vdims=['ytf'],label='new2').opts(color='red')
        b3ytfN = hv.Curve((self.band3['MHz']/1000,self.band3['ytf_new']),kdims=['GHz'],vdims=['ytf'],label='new3').opts(color='red')
        b4ytfN = hv.Curve((self.band4['MHz']/1000,self.band4['ytf_new']),kdims=['GHz'],vdims=['ytf'],label='new4').opts(color='red')
        b5ytfN = hv.Curve((self.band5['MHz']/1000,self.band5['ytf_new']),kdims=['GHz'],vdims=['ytf'],label='new5').opts(color='red')
        yc_new = (b1ytfN*b2ytfN*b3ytfN*b4ytfN*b5ytfN).opts(hv.opts.Curve(line_width=1,title='YTF Cal',tools=['hover'],width=1000,show_grid=True,xticks=10))
        return (yc_old*yc_new).opts(legend_position='top')
        

    def plt(self):
        return (self.plt_flatness() +  self.plt_ytf()).cols(1)