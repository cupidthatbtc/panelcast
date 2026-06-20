# Raw Schema Snapshot

Source
- `path/to/your/dataset.csv` (set via AOTY_DATASET_PATH environment variable)

Observed columns (first rows)
- Artist
- Album
- Year
- Release Date
- Genres
- Critic Score
- User Score
- Avg Track Score
- User Ratings
- Critic Reviews
- Tracks
- Runtime (min)
- Avg Track Runtime (min)
- Label
- Descriptors
- Album URL
- All Artists
- Album Type

Observed dtypes (sample)
- Artist: object
- Album: object
- Year: int64
- Release Date: object
- Genres: object
- Critic Score: float64
- User Score: int64
- Avg Track Score: float64
- User Ratings: int64
- Critic Reviews: float64
- Tracks: float64
- Runtime (min): float64
- Avg Track Runtime (min): float64
- Label: object
- Descriptors: float64
- Album URL: object
- All Artists: object
- Album Type: object

Notes
- Dtypes are from a small sample; full-file typing may differ.
- `Descriptors` appears numeric in the sample due to missing values.
