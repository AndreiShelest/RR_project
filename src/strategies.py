from backtesting import Strategy
from constants import signal_label


class TradingStrategy(Strategy):
    def init(self):
        super().init()

        if self.data[signal_label][0] == 1:
            self.buy()

    def next(self):
        curr_signal = self.data[signal_label][-1]
        prev_signal = self.data[signal_label][-2]

        if curr_signal == 1 and prev_signal == 0 and not self.position:
            self.buy()
        elif curr_signal == 0 and prev_signal == 1 and self.position:
            self.position.close()