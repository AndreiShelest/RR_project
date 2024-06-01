from backtesting import Strategy
from constants import signal_label


class TradingStrategy(Strategy):
    def init(self):
        super().init()

        if self.data[signal_label][0] == 1:
            self.buy()
        elif self.data[signal_label][0] == 0:
            self.sell()

    def next(self):
        curr_signal = self.data[signal_label][-1]
        prev_signal = self.data[signal_label][-2]

        if curr_signal == 1 and prev_signal == 0:
            if self.position:
                self.position.close()
            self.buy()
        elif curr_signal == 0 and prev_signal == 1:
            if self.position:
                self.position.close()
            self.sell()


class BuyAndHoldStrategy(Strategy):
    def init(self):
        super().init()
        self.buy()

    def next(self):
        super().next()
