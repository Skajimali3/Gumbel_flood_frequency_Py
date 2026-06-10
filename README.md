#Gumbel Flood Frequency Analysis in Python 3.11

This repository provides Python 3.11 scripts for flood frequency analysis using the Gumbel Extreme Value Type-I distribution. The workflow was developed to estimate transboundary flood frequency, peak discharge probability, and return periods for the Kosi and Teesta river basins.

Features:
Historical discharge preprocessing
Annual maximum series extraction
Gumbel distribution fitting
Flood return period estimation
Flood quantile calculation
Statistical visualization
Upstream–downstream comparison

Methodology:
The analysis follows the Gumbel Extreme Value framework:

[
F(x)=\exp\left[-\exp\left(-\frac{x-u}{\alpha}\right)\right]
]

Return period ((T)) is calculated as:

[
T = \frac{1}{P}
]

Flood magnitude for return period (T):

[
Q_T = \bar{Q} + K_T \sigma
]

where:

(Q_T) = flood discharge for return period (T)
(\bar{Q}) = mean annual maximum discharge
(\sigma) = standard deviation
(K_T) = Gumbel frequency factor

Data Source:
GEOGLOWS River Forecast System
Historical river discharge observations


Software Requirements:
Python 3.11
NumPy
Pandas
SciPy
Matplotlib
Seaborn


Applications:
Flood frequency analysis
Extreme flood estimation
Return period assessment
Climate change flood studies
Transboundary river basin management
