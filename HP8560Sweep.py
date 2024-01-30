from dataclasses import dataclass
import numpy as np, pandas as pd
import os, time
import scipy.signal
import pickle
import holoviews as hv
import tqdm

from HP8560SweepResults import HP8560SweepResults


class Lab:
    # This class proxies all communication to/from the instruments.
    # If you use python-vxi11, the interface is trivial. If not, it should be adaptable.
    #
    #
    # import vxi11
    #
    # sig_gen = vxi11.Instrument(host="ad007-right.lan", name="gpib0,19")
    # print(sig_gen.ask("*IDN?"))
    # power_meter = vxi11.Instrument(host="ad007-right.lan", name="gpib0,13")
    # print(power_meter.ask("*IDN?"))
    # spec_an = vxi11.Instrument(host="ad007-left.lan", name="gpib0,18")
    # print(f'{spec_an.ask("ID?")} {spec_an.ask("SER?")}')
    #
    # lab = Lab(sig_gen, power_meter, spec_an)
    def __init__(self, sig_gen, power_meter, spec_an):
        self.sig_gen = sig_gen
        self.power_meter = power_meter
        self.spec_an = spec_an
    def sig_gen_ask(self, command):
        return self.sig_gen.ask(command)
    def sig_gen_write(self, command):
        self.sig_gen.write(command)
    def power_meter_write(self, command):
        self.power_meter.write(command)
    def power_meter_ask(self, command):
        return self.power_meter.ask(command)
    def spec_an_write(self, command):
        self.spec_an.write(command)
    def spec_an_ask(self, command):
        return self.spec_an.ask(command)

@dataclass
class Band:
    name : str
    is_preselected : bool
    f_MHz : list[float]
    bump_MHz : float  # Normally in a band overlap the SA will default to LOWER band. Including this forces UPPER band.
    dac_sweep : list[int]


class HP8560Sweep:
    def __init__(self, ghz):
        assert(ghz in [2.9, 6.5, 13.2, 26.5, 40, 50])
        self.ghz = ghz
        bands= [
            Band(name='band0', is_preselected=False, f_MHz=np.concatenate(([2,6,10,27,44,61,78,95],np.arange(100,2950,50))), bump_MHz=None,       dac_sweep=None),
            Band(name='band1', is_preselected=True,  f_MHz=np.arange(2800,6450,50),                                          bump_MHz=2901,       dac_sweep=range(100,200)),
            Band(name='band2', is_preselected=True,  f_MHz=np.arange(5700,13250,50),                                         bump_MHz=6461,       dac_sweep=range(100,200)),
            Band(name='band3', is_preselected=True,  f_MHz=np.arange(12400,26950,50),                                        bump_MHz=13231,      dac_sweep=range(75 ,200)),
            Band(name='band4', is_preselected=True,  f_MHz=np.arange(26450,31250,200),                                       bump_MHz=26933,      dac_sweep=range(0,255,5)),
            Band(name='band5', is_preselected=True,  f_MHz=np.arange(31100,50100,200),                                       bump_MHz=31154,      dac_sweep=range(0,255,5))
        ]
        self.bands = [b for b in bands if (b.f_MHz[0]+b.f_MHz[-1])/2/1000 < ghz]
        self.name_to_band = {b.name:b for b in bands}
    def pts(self):
        pts = []
        for band in self.bands:
            for i,f_MHz in enumerate(band.f_MHz):
                fa_MHz = f_MHz - 5
                fb_MHz = f_MHz + 5
                bump = band.bump_MHz
                if bump and fb_MHz < band.bump_MHz:
                    fb_MHz = band.bump_MHz
                pts.append((f_MHz, fa_MHz, fb_MHz, band.name, i))
        return pts
    def band_after(self, band):
        i = self.bands.index(band)
        return self.bands[i+1] if i<(len(self.bands)-1) else None
    def band_before(self, band):
        i = self.bands.index(band)
        return self.bands[i-1] if i>0 else None
    def sweep(self, lab, ytf_scan=True):
        sg_min_hz = float(lab.sig_gen_ask(':FREQ:CW MIN\nFREQ?\n').strip())
        sg_max_hz = float(lab.sig_gen_ask(':FREQ:CW MAX\nFREQ?\n').strip())
        lab.power_meter_write("CFSRC A,FREQ")
        lab.power_meter_write("FROFF ON")
        lab.sig_gen_write('*RST')
        lab.sig_gen_write(":POWER:LEVEL 0.00")
        lab.sig_gen_write(":POWER:STATE ON")
        lab.sig_gen_write(":FREQ:MODE CW")
        pts = self.pts()
        fa,fb = 10e6, int(self.ghz*1e9)
        lab.spec_an_write(f'IP; FA {int(fa)}; FB {int(fb)}; RB 3MHZ; DET POS; SNGLS; TS; TS;')
        
        print(f"--- Beginning Scan ---")
        print("Full Span Sweeps")
        full_span_sweeps = []
        for f_MHz, fa_MHz, fb_MHz, band_name, i in tqdm.tqdm(pts):
            if f_MHz<10:
                f_MHz, fa_MHz, fb_MHz = 10, 5, 15
            f, fa, fb = f_MHz*1e6, fa_MHz*1e6, fb_MHz*1e6
            lab.sig_gen_write(f":FREQ:CW {int(f)}")
            lab.power_meter_write(f"CFFRQ A,{int(f)}")
            trace_freqs = np.linspace(fa,fb,601)
            time.sleep(.1)
            lab.spec_an_write(f"TS;")
            pm = float(lab.power_meter_ask("O 1").strip())
            tra = np.array([float(fs) for fs in lab.spec_an_ask('TRA?').split(',')])
            full_span_sweeps.append(dict(
                f_hz=f,
                power_meter_dBm=pm,
                trace_dB=tra,
                trace_f=trace_freqs,
                band=band_name,
                band_i=i
            ))
        
        print("Narrow Sweeps")
        lab.spec_an_write('IP; CF 1e9; SP 10MHZ; RB 3MHZ; DET POS; SNGLS; TS; TS;')
        narrow_sweeps = []
        for f_MHz, fa_MHz, fb_MHz, band_name, i in tqdm.tqdm(pts):
            if f_MHz<10:
                f_MHz, fa_MHz, fb_MHz = 10, 5, 15
            f, fa, fb = f_MHz*1e6, fa_MHz*1e6, fb_MHz*1e6
            lab.sig_gen_write(f":FREQ:CW {int(f)}")
            lab.power_meter_write(f"CFFRQ A,{int(f)}")
            trace_freqs = np.linspace(fa,fb,601)
            time.sleep(.1)
            lab.spec_an_write(f"FA {int(fa)}; FB {int(fb)}; TS; TS;")
            pm = float(lab.power_meter_ask("O 1").strip())
            tra = np.array([float(fs) for fs in lab.spec_an_ask('TRA?').split(',')])
            narrow_sweeps.append(dict(
                f_hz=f,
                power_meter_dBm=pm,
                trace_dB=tra,
                trace_f=trace_freqs,
                band=band_name,
                band_i=i
            ))
        
        if not ytf_scan:
            dac_sweeps = []
        if ytf_scan:
            print("YTF Sweeps")
            lab.spec_an_write('IP; CF 1e9; SP 10MHZ; RB 3MHZ; DET POS; SNGLS; TS; TS;')
            dac_sweeps = []
            for f_MHz, fa_MHz, fb_MHz, band_name, i in tqdm.tqdm(pts):
                if f_MHz<10:
                    f_MHz, fa_MHz, fb_MHz = 10, 5, 15
                f, fa, fb = f_MHz*1e6, fa_MHz*1e6, fb_MHz*1e6
                lab.sig_gen_write(f":FREQ:CW {int(f)}")
                lab.power_meter_write(f"CFFRQ A,{int(f)}")
                trace_freqs = np.linspace(fa,fb,601)
                time.sleep(.1)
                lab.spec_an_write(f"FA {int(fa)}; FB {int(fb)}; TS;")
                dac_sweep_dac = []
                dac_sweep_dBm = []
                band = self.name_to_band[band_name]
                dac_values_to_sweep = band.dac_sweep
                if not dac_values_to_sweep:
                    continue
                lab.spec_an_ask(f'PSDAC {dac_values_to_sweep[0]}; TS; MKPK HI; MKA?')
                for psdac in dac_values_to_sweep:
                    dBm = float(lab.spec_an_ask(f'PSDAC {psdac}; TS; MKPK HI; MKA?'))
                    dac_sweep_dac.append(psdac)
                    dac_sweep_dBm.append(dBm)
                max_dac = dac_sweep_dac[np.argmax(dac_sweep_dBm)]
                lab.spec_an_write(f'PSDAC {max_dac};')
                psdac = str(max_dac)
                        
                pm = float(lab.power_meter_ask("O 1").strip())
                tra = np.array([float(fs) for fs in lab.spec_an_ask('TS; TS; TRA?').split(',')])
                dac_sweeps.append(dict(
                    f_hz=f,
                    power_meter_dBm=pm,
                    trace_dB=tra,
                    trace_f=trace_freqs,
                    psdac=psdac,
                    band=band_name,
                    band_i=i,
                    dac_sweep_dac=dac_sweep_dac,
                    dac_sweep_dBm=dac_sweep_dBm
                ))

        return HP8560SweepResults(
            full_span_sweeps=full_span_sweeps,
            narrow_sweeps=narrow_sweeps,
            dac_sweeps=dac_sweeps
        )