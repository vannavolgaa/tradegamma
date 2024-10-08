from dataclasses import dataclass
from datetime import timedelta 

@dataclass
class BacktestParameters: 
    csv_lines_to_load: int = 1000000

    garch_time_delta: timedelta = timedelta(days = 10)

    auto_regressive_time_delta: timedelta = timedelta(days = 10)



