# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:45:44 2022

@author: cfai2
"""
import matplotlib

starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
import logging
from time import perf_counter
from functools import partial

from utils import normalize, LikelihoodData
from plotutils import PlotState
#import analyze_bayes
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'font.family':'STIXGeneral'})
matplotlib.rcParams.update({'mathtext.fontset':'stix'})


class ValidEntry(tk.ttk.Entry):
    """Entrybox that marks itself red when input is nonnumeric"""
    
    def __init__(self, root, master, width, textvariable=None):
        vcom = root.register(self.validate_is_numeric)
        super().__init__(master, width=width, textvariable=textvariable, validate='focusout', validatecommand=(vcom, '%P'))
                                          
    def validate_is_numeric(self, what):
        try:
            w = float(what)
            self.mark_error(0)
            return True
        except Exception:
            self.mark_error(1)
            return False
        
    def mark_error(self, err=0):
        if err:
            self['foreground'] = 'red'
        else:
            self['foreground'] = 'black'
        
class TkGUI():
    
    def __init__(self, title):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', False)
        self.root.title(title)
        
        s = ttk.Style()
        s.theme_use('classic')
        
        self.header_style = ttk.Style()
        self.header_style.configure("Header.TLabel", background='#D0FFFF',
                                    highlightbackground='#000000')
        
        self.plots = PlotState()
        self.data = LikelihoodData()
        
        ## TODO: These should go in a config file ##
        self.data.PARAM_ORDER = [r"$n_0$", r"$p_0$", r"$\mu_n$", r"$\mu_p$", 
                       r"$k^*$", r"$S_F$", r"$S_B$", r"$\tau_n$", 
                       r"$\tau_p$", r"$\lambda$", r"$m$", r"$\tau_{eff}$",
                       r"$\tau_{rad}$",r"$(S_F+S_B)$", r"$\mu\prime$", r"$\epsilon$", r"$\tau_n+\tau_p$"]
        
        seconds = [r"$\tau_{eff}$", r"$\tau_{rad}$", r"$(S_F+S_B)$", 
                                 r"$\mu\prime$", r"$\epsilon$", r"$\tau_n+\tau_p$"]
        
        self.data.SECONDARY_PARAMS = {param:(param in seconds) for param in self.data.PARAM_ORDER}
        
        self.plots.units = {r"$k^*$":r"cm$^{3}$ s$^{-1}$", r"$\mu_n$":r"cm$^{2}$ V$^{-1}$ s$^{-1}$", r"$\mu_p$":r"cm$^{2}$ V$^{-1}$ s$^{-1}$",
                          r"$S_F$":r"cm s$^{-1}$", r"$S_B$":r"cm $s^{-1}$", r"$n_0$":r"cm$^{-3}$", r"$p_0$":r"cm$^{-3}$", 
                          r"$\tau_n$":r"ns", r"$\tau_p$":r"ns",
                          r"$m$":"", r"$\epsilon$":"", r"$\tau_{eff}$":r"ns",
                          r"$\tau_{rad}$":r"ns", r"$(S_F+S_B)$":r"cm s$^{-1}$", r"$\mu\prime$":r"cm$^{2}$ V$^{-1}$ s$^{-1}$",
                          r"$\tau_n+\tau_p$":"ns"}
        self.has_fig=False
        ############################################
        
        self.make_notebook()
        
        return
    
    def run(self):
        width, height = self.root.winfo_screenwidth() * 0.6, self.root.winfo_screenheight() * 0.8

        self.root.geometry('%dx%d+0+0' % (width,height))
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes,'-topmost',False)
        self.root.mainloop()
        self.quit()
        return
    
    def quit(self):
        self.update_config_file()        
        matplotlib.use(starting_backend)
        return
    
    def enter(self, entryBox, text):
        """ Fill user entry boxes with text. """
        entryBox.delete(0,tk.END)
        entryBox.insert(0,text)
        return

    
    def make_notebook(self):
        self.main_canvas = tk.Canvas(self.root)
        self.main_canvas.grid(row=0,column=0, sticky='nswe')
        
        self.notebook = tk.ttk.Notebook(self.main_canvas)
        
        # Allocate room for and add scrollbars to overall notebook
        self.main_scroll_y = tk.ttk.Scrollbar(self.root, orient="vertical", 
                                              command=self.main_canvas.yview)
        self.main_scroll_y.grid(row=0,column=1, sticky='ns')
        self.main_scroll_x = tk.ttk.Scrollbar(self.root, orient="horizontal", 
                                              command=self.main_canvas.xview)
        self.main_scroll_x.grid(row=1,column=0,sticky='ew')
        self.main_canvas.configure(yscrollcommand=self.main_scroll_y.set, 
                                   xscrollcommand=self.main_scroll_x.set)
        # Make area for scrollbars as narrow as possible without cutting off
        self.root.rowconfigure(0,weight=100)
        self.root.rowconfigure(1,weight=1, minsize=20) 
        self.root.columnconfigure(0,weight=100)
        self.root.columnconfigure(1,weight=1, minsize=20)
        
        self.main_canvas.create_window((0,0), window=self.notebook, anchor="nw")
        self.notebook.bind('<Configure>', 
                           lambda e:self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all')))
    
        self.menu_bar = tk.Menu(self.notebook)
    
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Exit", 
                                   command=self.root.destroy)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        
        self.options_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.options_menu.add_command(label="Parameter Options", 
                                   command=self.set_param_options)
        self.options_menu.add_command(label="Plot Options", 
                                   command=self.set_plot_options)
        self.menu_bar.add_cascade(label="Settings", menu=self.options_menu)
        
        self.root.config(menu=self.menu_bar)
        
        
        self.make_frame()
        self.load_config_file()
    
    def make_frame(self):
        self.main_frame = tk.ttk.Frame(self.notebook)
        
        self.control_frame = tk.ttk.Frame(self.main_frame)
        self.control_frame.grid(row=0,column=0)
        
        tk.ttk.Button(self.control_frame, text="Enable/Disable Parameters", 
                      command=self.toggle_params).grid(row=0,column=0)
        
        self.IO_frame = tk.ttk.Frame(self.main_frame)
        self.IO_frame.grid(row=1,column=0)
        
        tk.ttk.Button(self.IO_frame, text="Load", 
                      command=self.plot).grid(row=0,column=0)
        
        self.analysis_frame = tk.ttk.Frame(self.main_frame)
        self.analysis_frame.grid(row=2, column=0)
        
        tk.ttk.Button(self.analysis_frame, text=r"Stats Report",
                      command=self.calc_stats_report).grid(row=0,column=0)
        
        tk.ttk.Button(self.analysis_frame, text=r"Find max uncertainty",
                      command=self.calc_max_uncertainty).grid(row=1,column=0)
        
        tk.ttk.Button(self.analysis_frame, text=r"Show covariance matrix", 
                      command=self.show_covariance).grid(row=2, column=0)
        
        self.plot_frame = tk.ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=0,column=1, rowspan=99)
        
        self.setup_plot2D(2)
        self.notebook.add(self.main_frame, text="Main")
        
    def setup_plot2D(self, num_plots):
        if self.has_fig:
            self.fig_canvas.get_tk_widget().destroy()
            self.fig_toolbar_frame.destroy()
        
        self.plots.setup_2D(num_plots)
        
        self.fig_canvas = tkagg.FigureCanvasTkAgg(self.plots.fig, 
                                                         master=self.plot_frame)
        self.fig_canvas.get_tk_widget().grid(row=0,column=0)
        
        self.fig_toolbar_frame = tk.ttk.Frame(master=self.plot_frame)
        self.fig_toolbar_frame.grid(row=1,column=0)
        tkagg.NavigationToolbar2Tk(self.fig_canvas, 
                                   self.fig_toolbar_frame)
        self.has_fig=True
        
    def init_param_options_panel(self, scales, limits, thickness, num_obs):
        """ Default values, axis limits, etc for parameter display options"""

        self.param_dolog = {}
        for r, param in enumerate(self.data.PARAM_ORDER):
            v = scales.get(param, 0)
            self.param_dolog[param] = tk.IntVar(value=v)
            
        self.param_limits = {}
        
        for r, param in enumerate(self.data.PARAM_ORDER):
            
            if param in limits:
                self.param_limits[param] = [tk.StringVar(value=limits[param][0]), 
                                            tk.StringVar(value=limits[param][1])]
                
            else:
                self.param_limits[param] = [tk.StringVar(value=""), 
                                            tk.StringVar(value="")]
                
        self.thickness = tk.StringVar(value=thickness)
        self.num_observations = tk.StringVar(value=num_obs)
                
        return
        
    def make_param_options_panel(self):
        """ Update values, axis limits, etc """
        self.param_options_panel = tk.Toplevel(self.root)
        
        params_frame = tk.ttk.Frame(self.param_options_panel)
        params_frame.grid(row=0,column=0,columnspan=99)

        header = ("Parameter", "Log scale?", "Lower bound", "Upper bound")

        for c, h in enumerate(header):    
            ttk.Label(params_frame, text=h, style="Header.TLabel").grid(row=0,column=c)

        param_l_limit_entries = {}
        param_u_limit_entries = {}
        for r, param in enumerate(self.data.PARAM_ORDER):
            tk.Label(params_frame, text=r"{}".format(param)).grid(row=r+1, column=0)
            tk.ttk.Checkbutton(params_frame, variable=self.param_dolog[param]).grid(row=r+1, column=1)

        for r, param in enumerate(self.data.PARAM_ORDER):
            if not self.data.SECONDARY_PARAMS[param]:
                param_l_limit_entries[param] = ValidEntry(self.root, params_frame, width=10,
                                                          textvariable=self.param_limits[param][0])
                param_u_limit_entries[param] = ValidEntry(self.root, params_frame, width=10,
                                                          textvariable=self.param_limits[param][1])
                param_l_limit_entries[param].grid(row=r+1, column=2)
                param_u_limit_entries[param].grid(row=r+1, column=3)
                
        for entry in param_l_limit_entries:
            self.enter(param_l_limit_entries[entry], self.param_limits[entry][0].get())
            self.enter(param_u_limit_entries[entry], self.param_limits[entry][1].get())

        ttk.Label(self.param_options_panel, text="Thickness [nm]").grid(row=1,column=0)
        
        ValidEntry(self.root, self.param_options_panel, width=10, textvariable=self.thickness).grid(row=1,column=1)
        
        ttk.Label(self.param_options_panel, text="Num. Observations").grid(row=2,column=0)
        
        ValidEntry(self.root, self.param_options_panel, width=10, textvariable=self.num_observations).grid(row=2,column=1)
        
        self.param_options_panel.protocol("WM_DELETE_WINDOW", self.on_param_options_panel_close)
        self.param_options_panel.grab_set()
        return
    
    def on_param_options_panel_close(self):
        self.param_options_panel.destroy()
        return
        
    def set_param_options(self, from_file=False):
        self.make_param_options_panel()
        self.root.wait_window(self.param_options_panel)
        return
    
    def init_paramtoggles(self, enabled, marks):
        self.param_enabled = {}
        for r, param in enumerate(self.data.PARAM_ORDER):
            v = enabled.get(param, 0)
            self.param_enabled[param] = tk.IntVar(value=v)
            
        self.param_marked_values = {}
        for r, param in enumerate(self.data.PARAM_ORDER):
            if param in marks:
                v = marks[param]
            else:
                v = ""
            self.param_marked_values[param] = tk.StringVar(value=v)
        
        return
    
    def make_paramtoggles(self):
        self.paramtoggles = tk.Toplevel(self.root)
        
        header = ("Parameter", "Enable", "Mark value")

        for c, h in enumerate(header):    
            ttk.Label(self.paramtoggles, text=h, style="Header.TLabel").grid(row=0,column=c)

        for r, param in enumerate(self.data.PARAM_ORDER):
            tk.Label(self.paramtoggles, text=r"{}".format(param)).grid(row=r+1, column=0)
            tk.ttk.Checkbutton(self.paramtoggles, variable=self.param_enabled[param]).grid(row=r+1, column=1)

        self.param_marked_entries = {}
        for r, param in enumerate(self.data.PARAM_ORDER):
            self.param_marked_entries[param] = ValidEntry(self.root, self.paramtoggles, 
                                                          width=10, textvariable=self.param_marked_values[param])
            self.param_marked_entries[param].grid(row=r+1, column=2)
                
        for entry in self.param_marked_entries:
            self.enter(self.param_marked_entries[entry], self.param_marked_values[entry].get())

        self.paramtoggles.protocol("WM_DELETE_WINDOW", self.on_paramtoggles_close)
        self.paramtoggles.grab_set()
        
        return
    
    def on_paramtoggles_close(self):
        self.paramtoggles.destroy()
        return
    
    def toggle_params(self):
        self.make_paramtoggles()
        self.root.wait_window(self.paramtoggles)
        
    def init_plot_options(self):
        self.bin_count = tk.StringVar(value=96)
        self.num_maxes = tk.StringVar(value=50)
        self.c_value = tk.StringVar(value=(1/2000))
        self.exclude_outside_limits = tk.IntVar(value=1)
        return
        
    def make_plot_options(self):
        self.plot_options = tk.Toplevel(self.root)
        
        ttk.Label(self.plot_options, text="Num. Bins", style="Header.TLabel").grid(row=0,column=0)
        ValidEntry(self.root, self.plot_options, width=10, textvariable=self.bin_count).grid(row=0,column=1)
        
        ttk.Label(self.plot_options, text="Num. Maxes", style="Header.TLabel").grid(row=1,column=0)
        ValidEntry(self.root, self.plot_options, width=10, textvariable=self.num_maxes).grid(row=1,column=1)
        
        ttk.Label(self.plot_options, text="C. Value", style="Header.TLabel").grid(row=2,column=0)
        ValidEntry(self.root, self.plot_options, width=10, textvariable=self.c_value).grid(row=2,column=1)
        
        ttk.Label(self.plot_options, text="Exclude outside limits?", style="Header.TLabel").grid(row=3,column=0)
        ttk.Checkbutton(self.plot_options,variable=self.exclude_outside_limits).grid(row=3,column=1)
        
        self.plot_options.protocol("WM_DELETE_WINDOW", self.on_plot_options_close)
        self.plot_options.grab_set()

    def on_plot_options_close(self):
        self.plot_options.destroy()
        return
    
    def set_plot_options(self):
        self.make_plot_options()
        self.root.wait_window(self.plot_options)
        
    def make_message_popup(self, msg):
        text_popup = tk.Toplevel(master=self.root)
        
        textbox = tk.ttk.Label(text_popup, text=msg)
        textbox.grid(row=0, column=0)
        
        tk.Button(text_popup, text="Continue", 
                  command=partial(self.on_popup_close, text_popup)).grid(row=1, column=0)
        
        text_popup.protocol("WM_DELETE_WINDOW", partial(self.on_popup_close, text_popup))
        
    def on_popup_close(self, popup):
        popup.destroy()
        
    def setup_covariance_plot(self):            
        cov_popup = tk.Toplevel(self.root)
        
        self.plots.setup_cov()
        
        self.cov_canvas = tkagg.FigureCanvasTkAgg(self.plots.cov_fig, 
                                                         master=cov_popup)
        self.cov_canvas.get_tk_widget().grid(row=0,column=0)

        tk.Button(cov_popup, text="Continue", 
                  command=partial(self.on_popup_close, cov_popup)).grid(row=1, column=0)
        
        cov_popup.protocol("WM_DELETE_WINDOW", partial(self.on_popup_close, cov_popup))
        
    def update_config_file(self):
        with open("config.txt", "w+") as ofstream:
            ofstream.write("!Parameter Scales\n")
            for param in self.param_dolog:
                ofstream.write("{}\t{}\n".format(param, self.param_dolog[param].get()))
                
            ofstream.write("!Parameter Ranges\n")
            for param in self.param_limits:
                if not self.data.SECONDARY_PARAMS[param]:
                    ofstream.write("{}\t{}\t{}\n".format(param, self.param_limits[param][0].get(), self.param_limits[param][1].get()))
                    
            ofstream.write("!Thickness\t{}\n".format(self.thickness.get()))
            ofstream.write("!Num_observations\t{}\n".format(self.num_observations.get()))
            
            ofstream.write("!Parameter Enabled\n")
            for param in self.param_enabled:
                ofstream.write("{}\t{}\n".format(param, self.param_enabled[param].get()))
                
            ofstream.write("!Parameter Marks\n")
            for param in self.param_marked_values:
                ofstream.write("{}\t{}\n".format(param, self.param_marked_values[param].get()))
        return
    
    def load_config_file(self, fname="config.txt"):
        mode = ''
        scales = {}
        limits = {}
        enabled = {}
        marks = {}
        thickness = 2000
        num_obs = 480000
        with open(fname, "r") as ifstream:
            for line in ifstream:
                line = line.strip(" \n")
                if line == "!Parameter Scales":
                    mode = 's'
                    continue
                
                elif line == "!Parameter Ranges":
                    mode = 'r'
                    continue
                
                elif line == "!Parameter Enabled":
                    mode = 'e'
                    continue
                
                elif line == "!Parameter Marks":
                    mode = 'm'
                    continue
                
                elif "!Thickness" in line:
                    try:
                        thickness = float(line.split("\t")[1])
                    except Exception:
                        continue
                    
                elif "!Num_observations" in line:
                    try:
                        num_obs = float(line.split("\t")[1])
                    except Exception:
                        continue
                
                if mode == 's':
                    try:
                        param, do_log = line.split("\t")
                        scales[param] = int(do_log)
                    except Exception:
                        continue
                    
                elif mode == 'r':
                    try:
                        param, lower, upper = line.split("\t")
                        limits[param] = (float(lower), float(upper))
                    except Exception:
                        continue
                    
                elif mode == 'e':
                    try:
                        param, e = line.split('\t')
                        enabled[param] = int(e)
                    except Exception:
                        continue
                    
                elif mode == 'm':
                    try:
                        param, mark = line.split('\t')
                        marks[param] = float(mark)
                    except Exception:
                        continue
                    
        self.init_param_options_panel(scales, limits, thickness, num_obs)
        self.init_paramtoggles(enabled, marks)
        self.init_plot_options()
        
    def load(self):
        ll_fname = tk.filedialog.askopenfilename(initialdir=".", 
                                                  title="Select a likelihood file", 
                                                  filetypes=[("Bayes likelihood files","*BAYRAN*.npy")])
        if not ll_fname: 
            raise Exception("No filename selected")
        
        self.data.load(ll_fname)
        return
    
    def plot(self):
        # Get data
        try:
            start = perf_counter()
            self.load()
            end = perf_counter()
            print("Load took {} s".format(end-start))
        except Exception:
            logging.warning("could not load selected likelihood file")
            return

        # Get limits, etc per parameter
        try:
            self.plots.fromTK_get_axis_limits(self.data, self.param_limits)
        except Exception:
            logging.error("invalid axis limits")
            return

        self.plots.enabled_params = [param for param in self.param_enabled if self.param_enabled[param].get()]
        
        if self.exclude_outside_limits.get():
            start = perf_counter()
            # Drop points outside of the rectangular region bounded by the param_limits
            self.data.exclude_using_axis_limits(self.plots)
            end = perf_counter()
            print("Exclusion took {} s".format(end-start)) 
        
        try:
            maxes = int(float((self.num_maxes.get())))
            if maxes < 0: maxes = 0
        except Exception:
            maxes = 0    
        
        self.plots.generate_P_ranking(self.data, maxes)
            
        # Pack into param-indexable dict
        self.data.pack_X_param_indexable()
                
        # Get thickness
        try:
            self.data.thickness = float(self.thickness.get())
            
            if self.data.thickness <= 0: raise ValueError("Invalid thickness")
        except Exception:
            logging.error("invalid thickness")
            return
        
        # Add in secondary parameters
        self.data.calculate_secondary_params(self.plots.enabled_params)
        self.plots.update_secondary_axis_limits(self.data)
            
        # Keep only the X we're about to marginalize
        # We only needed all the X earlier in case of secondary params
        self.data.strip_unneeded_params(self.plots.enabled_params)
        
        # Add values to be marked in red
        self.plots.add_mark_vals(self.param_marked_values)
        
        # Log scale conversion
        self.plots.do_logscale = {}
        for param in self.plots.enabled_params:
            self.plots.do_logscale[param] = self.param_dolog[param].get()
            
        self.data.logX(self.plots)
        self.plots.loglimits()
        
        # Get bin count 
        # TODO: validate when entered
        try:
            self.plots.bin_count = float(self.bin_count.get())
            
            if self.plots.bin_count < 2: raise ValueError("Too few bins")
        except Exception:
            logging.error("invalid number of bins")
            return        

        try:
            self.data.num_observations = float(self.num_observations.get())
            if self.data.num_observations <= 0: raise ValueError("invalid number of observations")
        except Exception:
            logging.error("invalid number of observations")
            return
        
        try:
            c =  float(self.c_value.get())
            if c <= 0: raise ValueError("Invalid c value")
        except Exception:
            logging.error("invalid c value")
            
        tf = self.data.num_observations * c
        self.data.P = self.data.LL / tf
        self.data.P = normalize(self.data.P)
        print(tf)
        
        self.setup_plot2D(len(self.plots.enabled_params))
        matplotlib.rcParams.update({'font.size': 8})

        #self.fig.suptitle(f"{key} tol=7 N={len(P)} T={tf}")
        
        start = perf_counter()
        self.data.marginalize1D(self.plots)
        end = perf_counter()
        print("1D Mar took {} s".format(end-start))
        
        start = perf_counter()
        self.data.marginalize2D(self.plots)
        end = perf_counter()
        print("2D Mar took {} s".format(end-start))

        self.plots.make_2D_grid(self.data)
        # import numpy as np
        # np.save("TEST_MUA", self.data.X[r"$\mu\prime$"])
        
    def calc_stats_report(self):
        means_and_stds = self.data.stats_summarize()
        msg = []
        for param in means_and_stds:
            msg.append("{}:".format(param))
            msg.append("Mean: {:.3e}".format(means_and_stds[param][0]))
            msg.append("Weighted sample std: {:.3e}".format(means_and_stds[param][1]))
            msg.append("W^2 Factor: {:.3e}".format(means_and_stds[param][2]))
            
        self.make_message_popup("\n".join(msg))
        
    def calc_max_uncertainty(self):
        uncertainty = self.data.calc_max_uncertainty()
        msg = []
        for param in uncertainty:
            msg.append("{}:".format(param))
            msg.append("Max uncertainty: {}".format(uncertainty[param][1]))
            msg.append("Best C value: {:.3e}".format(uncertainty[param][0] / self.data.num_observations))
            
        self.make_message_popup("\n".join(msg))
        
    def show_covariance(self):
        self.setup_covariance_plot()
        self.plots.heatmap_cov(self.data.calc_covariance(self.plots))
    
if __name__ == "__main__":
    nb = TkGUI("Marginalization")
    nb.run()