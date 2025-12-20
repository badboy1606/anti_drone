#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Jammer Gen
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from xmlrpc.server import SimpleXMLRPCServer
import threading
import osmosdr
import time



class jammer_gen(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Jammer Gen", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Jammer Gen")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "jammer_gen")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.var_rf_gain = var_rf_gain = 10
        self.var_if_gain = var_if_gain = 10
        self.var_cent_freq = var_cent_freq = 1.57542e9
        self.var_bb_gain = var_bb_gain = 10
        self.var_bandwidth = var_bandwidth = 10e6
        self.samp_rate = samp_rate = 10e6
        self.rf_gain = rf_gain = var_rf_gain
        self.if_gain = if_gain = var_if_gain
        self.cent_freq = cent_freq = var_cent_freq
        self.bb_gain = bb_gain = var_bb_gain
        self.bandwidth = bandwidth = var_bandwidth

        ##################################################
        # Blocks
        ##################################################

        self._rf_gain_range = qtgui.Range(00, 60, 10, var_rf_gain, 200)
        self._rf_gain_win = qtgui.RangeWidget(self._rf_gain_range, self.set_rf_gain, "RF gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rf_gain_win)
        self._if_gain_range = qtgui.Range(10, 60, 10, var_if_gain, 200)
        self._if_gain_win = qtgui.RangeWidget(self._if_gain_range, self.set_if_gain, "IF gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._if_gain_win)
        self._cent_freq_range = qtgui.Range(900e6, 6500e6, 500, var_cent_freq, 200)
        self._cent_freq_win = qtgui.RangeWidget(self._cent_freq_range, self.set_cent_freq, "Freq", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._cent_freq_win)
        self._bb_gain_range = qtgui.Range(10, 60, 10, var_bb_gain, 200)
        self._bb_gain_win = qtgui.RangeWidget(self._bb_gain_range, self.set_bb_gain, "BB gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._bb_gain_win)
        self._bandwidth_range = qtgui.Range(2e6, 50e6, 10, var_bandwidth, 200)
        self._bandwidth_win = qtgui.RangeWidget(self._bandwidth_range, self.set_bandwidth, "Bandwidth", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._bandwidth_win)
        self.xmlrpc_server_0 = SimpleXMLRPCServer(('localhost', 8888), allow_none=True)
        self.xmlrpc_server_0.register_instance(self)
        self.xmlrpc_server_0_thread = threading.Thread(target=self.xmlrpc_server_0.serve_forever)
        self.xmlrpc_server_0_thread.daemon = True
        self.xmlrpc_server_0_thread.start()
        self.osmosdr_sink_0 = osmosdr.sink(
            args="numchan=" + str(1) + " " + 'hackrf=0'
        )
        self.osmosdr_sink_0.set_sample_rate(10e6)
        self.osmosdr_sink_0.set_center_freq(cent_freq, 0)
        self.osmosdr_sink_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0.set_gain(rf_gain, 0)
        self.osmosdr_sink_0.set_if_gain(if_gain, 0)
        self.osmosdr_sink_0.set_bb_gain(bb_gain, 0)
        self.osmosdr_sink_0.set_antenna('', 0)
        self.osmosdr_sink_0.set_bandwidth(bandwidth, 0)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(100000)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 50, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.osmosdr_sink_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "jammer_gen")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_var_rf_gain(self):
        return self.var_rf_gain

    def set_var_rf_gain(self, var_rf_gain):
        self.var_rf_gain = var_rf_gain
        self.set_rf_gain(self.var_rf_gain)

    def get_var_if_gain(self):
        return self.var_if_gain

    def set_var_if_gain(self, var_if_gain):
        self.var_if_gain = var_if_gain
        self.set_if_gain(self.var_if_gain)

    def get_var_cent_freq(self):
        return self.var_cent_freq

    def set_var_cent_freq(self, var_cent_freq):
        self.var_cent_freq = var_cent_freq
        self.set_cent_freq(self.var_cent_freq)

    def get_var_bb_gain(self):
        return self.var_bb_gain

    def set_var_bb_gain(self, var_bb_gain):
        self.var_bb_gain = var_bb_gain
        self.set_bb_gain(self.var_bb_gain)

    def get_var_bandwidth(self):
        return self.var_bandwidth

    def set_var_bandwidth(self, var_bandwidth):
        self.var_bandwidth = var_bandwidth
        self.set_bandwidth(self.var_bandwidth)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self.osmosdr_sink_0.set_gain(self.rf_gain, 0)

    def get_if_gain(self):
        return self.if_gain

    def set_if_gain(self, if_gain):
        self.if_gain = if_gain
        self.osmosdr_sink_0.set_if_gain(self.if_gain, 0)

    def get_cent_freq(self):
        return self.cent_freq

    def set_cent_freq(self, cent_freq):
        self.cent_freq = cent_freq
        self.osmosdr_sink_0.set_center_freq(self.cent_freq, 0)

    def get_bb_gain(self):
        return self.bb_gain

    def set_bb_gain(self, bb_gain):
        self.bb_gain = bb_gain
        self.osmosdr_sink_0.set_bb_gain(self.bb_gain, 0)

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.osmosdr_sink_0.set_bandwidth(self.bandwidth, 0)




def main(top_block_cls=jammer_gen, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
