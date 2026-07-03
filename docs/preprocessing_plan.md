Missing timestamps

↓

Merge weather + AQ

↓

Interpolation

↓

Outlier removal

↓

Normalization

↓

Sliding window generation
## Incoming preprocessing

Weather

- Timestamp normalization
- Missing value detection

Air Quality

- Remove invalid measurements
- Handle missing PM2.5
- Merge monthly downloads
- Yearly CSV generation
- Station consistency checks

Combined

- Weather/AQ merge
- Time alignment
- Missing timestamp filling