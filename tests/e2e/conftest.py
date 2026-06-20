"""E2E test fixtures and setup for pipeline testing.

Provides fixtures for end-to-end testing of the AOTY prediction pipeline:
- Minimal test data with valid structure
- Temporary directories for isolated test runs
- Pipeline configurations for fast test execution
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from panelcast.pipelines.orchestrator import PipelineConfig

# Minimal test data: 3 artists with 3+ albums each (24 rows total)
# Required for artist history features (leave-one-out needs history)
MINIMAL_TEST_DATA = [
    # Artist 1: 4 albums
    {
        "Artist": "Test Artist One",
        "Album": "Debut Album",
        "Year": 2015,
        "Release Date": "January 15, 2015",
        "Genres": "Rock, Alternative",
        "Critic Score": 75,
        "User Score": 78,
        "Avg Track Score": 76,
        "User Ratings": 500,
        "Critic Reviews": 12,
        "Tracks": 10,
        "Runtime (min)": 42.5,
        "Avg Track Runtime (min)": 4.25,
        "Label": "Indie Records",
        "Descriptors": "melodic, guitar",
        "Album URL": "https://example.com/1",
        "All Artists": "Test Artist One",
        "Album Type": "Album",
    },
    {
        "Artist": "Test Artist One",
        "Album": "Sophomore",
        "Year": 2017,
        "Release Date": "March 10, 2017",
        "Genres": "Rock, Alternative",
        "Critic Score": 80,
        "User Score": 82,
        "Avg Track Score": 79,
        "User Ratings": 750,
        "Critic Reviews": 15,
        "Tracks": 12,
        "Runtime (min)": 48.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "Indie Records",
        "Descriptors": "energetic, rock",
        "Album URL": "https://example.com/2",
        "All Artists": "Test Artist One",
        "Album Type": "Album",
    },
    {
        "Artist": "Test Artist One",
        "Album": "Third Record",
        "Year": 2019,
        "Release Date": "July 22, 2019",
        "Genres": "Rock, Indie Rock",
        "Critic Score": 85,
        "User Score": 87,
        "Avg Track Score": 84,
        "User Ratings": 1200,
        "Critic Reviews": 20,
        "Tracks": 11,
        "Runtime (min)": 44.5,
        "Avg Track Runtime (min)": 4.05,
        "Label": "Major Records",
        "Descriptors": "mature, layered",
        "Album URL": "https://example.com/3",
        "All Artists": "Test Artist One",
        "Album Type": "Album",
    },
    {
        "Artist": "Test Artist One",
        "Album": "Latest Release",
        "Year": 2022,
        "Release Date": "November 5, 2022",
        "Genres": "Rock, Indie Rock",
        "Critic Score": 82,
        "User Score": 85,
        "Avg Track Score": 83,
        "User Ratings": 900,
        "Critic Reviews": 18,
        "Tracks": 10,
        "Runtime (min)": 40.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "Major Records",
        "Descriptors": "refined",
        "Album URL": "https://example.com/4",
        "All Artists": "Test Artist One",
        "Album Type": "Album",
    },
    # Artist 2: 4 albums (different genre)
    {
        "Artist": "Electronic Producer",
        "Album": "Digital Debut",
        "Year": 2016,
        "Release Date": "February 20, 2016",
        "Genres": "Electronic, Ambient",
        "Critic Score": 70,
        "User Score": 72,
        "Avg Track Score": 71,
        "User Ratings": 300,
        "Critic Reviews": 8,
        "Tracks": 8,
        "Runtime (min)": 52.0,
        "Avg Track Runtime (min)": 6.5,
        "Label": "Electronic Label",
        "Descriptors": "atmospheric",
        "Album URL": "https://example.com/5",
        "All Artists": "Electronic Producer",
        "Album Type": "Album",
    },
    {
        "Artist": "Electronic Producer",
        "Album": "Second Wave",
        "Year": 2018,
        "Release Date": "June 15, 2018",
        "Genres": "Electronic, IDM",
        "Critic Score": 78,
        "User Score": 80,
        "Avg Track Score": 77,
        "User Ratings": 450,
        "Critic Reviews": 10,
        "Tracks": 9,
        "Runtime (min)": 55.0,
        "Avg Track Runtime (min)": 6.1,
        "Label": "Electronic Label",
        "Descriptors": "complex, glitchy",
        "Album URL": "https://example.com/6",
        "All Artists": "Electronic Producer",
        "Album Type": "Album",
    },
    {
        "Artist": "Electronic Producer",
        "Album": "Third Chapter",
        "Year": 2020,
        "Release Date": "September 1, 2020",
        "Genres": "Electronic, Experimental",
        "Critic Score": 83,
        "User Score": 81,
        "Avg Track Score": 80,
        "User Ratings": 600,
        "Critic Reviews": 14,
        "Tracks": 10,
        "Runtime (min)": 58.0,
        "Avg Track Runtime (min)": 5.8,
        "Label": "Major Electronic",
        "Descriptors": "bold, innovative",
        "Album URL": "https://example.com/7",
        "All Artists": "Electronic Producer",
        "Album Type": "Album",
    },
    {
        "Artist": "Electronic Producer",
        "Album": "Current Work",
        "Year": 2023,
        "Release Date": "April 12, 2023",
        "Genres": "Electronic, Ambient",
        "Critic Score": 79,
        "User Score": 82,
        "Avg Track Score": 79,
        "User Ratings": 550,
        "Critic Reviews": 12,
        "Tracks": 8,
        "Runtime (min)": 50.0,
        "Avg Track Runtime (min)": 6.25,
        "Label": "Major Electronic",
        "Descriptors": "meditative",
        "Album URL": "https://example.com/8",
        "All Artists": "Electronic Producer",
        "Album Type": "Album",
    },
    # Artist 3: 4 albums (hip-hop)
    {
        "Artist": "Hip Hop Artist",
        "Album": "First Bars",
        "Year": 2014,
        "Release Date": "October 8, 2014",
        "Genres": "Hip Hop, Conscious Hip Hop",
        "Critic Score": 72,
        "User Score": 75,
        "Avg Track Score": 73,
        "User Ratings": 400,
        "Critic Reviews": 9,
        "Tracks": 14,
        "Runtime (min)": 56.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "Hip Hop Records",
        "Descriptors": "lyrical, raw",
        "Album URL": "https://example.com/9",
        "All Artists": "Hip Hop Artist",
        "Album Type": "Album",
    },
    {
        "Artist": "Hip Hop Artist",
        "Album": "Growth",
        "Year": 2016,
        "Release Date": "May 25, 2016",
        "Genres": "Hip Hop, Jazz Rap",
        "Critic Score": 82,
        "User Score": 85,
        "Avg Track Score": 83,
        "User Ratings": 800,
        "Critic Reviews": 16,
        "Tracks": 12,
        "Runtime (min)": 50.0,
        "Avg Track Runtime (min)": 4.17,
        "Label": "Major Hip Hop",
        "Descriptors": "jazzy, conscious",
        "Album URL": "https://example.com/10",
        "All Artists": "Hip Hop Artist",
        "Album Type": "Album",
    },
    {
        "Artist": "Hip Hop Artist",
        "Album": "Peak",
        "Year": 2018,
        "Release Date": "December 1, 2018",
        "Genres": "Hip Hop, Jazz Rap",
        "Critic Score": 90,
        "User Score": 92,
        "Avg Track Score": 89,
        "User Ratings": 2000,
        "Critic Reviews": 25,
        "Tracks": 13,
        "Runtime (min)": 52.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "Major Hip Hop",
        "Descriptors": "masterpiece, dense",
        "Album URL": "https://example.com/11",
        "All Artists": "Hip Hop Artist",
        "Album Type": "Album",
    },
    {
        "Artist": "Hip Hop Artist",
        "Album": "Recent Work",
        "Year": 2021,
        "Release Date": "August 20, 2021",
        "Genres": "Hip Hop, Experimental Hip Hop",
        "Critic Score": 84,
        "User Score": 86,
        "Avg Track Score": 84,
        "User Ratings": 1500,
        "Critic Reviews": 20,
        "Tracks": 11,
        "Runtime (min)": 45.0,
        "Avg Track Runtime (min)": 4.09,
        "Label": "Major Hip Hop",
        "Descriptors": "experimental, bold",
        "Album URL": "https://example.com/12",
        "All Artists": "Hip Hop Artist",
        "Album Type": "Album",
    },
    # Artist 4: 3 albums (pop - ensures variety)
    {
        "Artist": "Pop Singer",
        "Album": "Pop Debut",
        "Year": 2017,
        "Release Date": "April 5, 2017",
        "Genres": "Pop, Dance Pop",
        "Critic Score": 65,
        "User Score": 70,
        "Avg Track Score": 68,
        "User Ratings": 200,
        "Critic Reviews": 6,
        "Tracks": 12,
        "Runtime (min)": 38.0,
        "Avg Track Runtime (min)": 3.17,
        "Label": "Pop Records",
        "Descriptors": "catchy, fun",
        "Album URL": "https://example.com/13",
        "All Artists": "Pop Singer",
        "Album Type": "Album",
    },
    {
        "Artist": "Pop Singer",
        "Album": "Pop Evolution",
        "Year": 2019,
        "Release Date": "August 15, 2019",
        "Genres": "Pop, Synth Pop",
        "Critic Score": 72,
        "User Score": 76,
        "Avg Track Score": 74,
        "User Ratings": 350,
        "Critic Reviews": 10,
        "Tracks": 10,
        "Runtime (min)": 35.0,
        "Avg Track Runtime (min)": 3.5,
        "Label": "Major Pop",
        "Descriptors": "polished, synthy",
        "Album URL": "https://example.com/14",
        "All Artists": "Pop Singer",
        "Album Type": "Album",
    },
    {
        "Artist": "Pop Singer",
        "Album": "Pop Maturity",
        "Year": 2022,
        "Release Date": "January 20, 2022",
        "Genres": "Pop, Art Pop",
        "Critic Score": 78,
        "User Score": 80,
        "Avg Track Score": 77,
        "User Ratings": 500,
        "Critic Reviews": 14,
        "Tracks": 11,
        "Runtime (min)": 40.0,
        "Avg Track Runtime (min)": 3.64,
        "Label": "Major Pop",
        "Descriptors": "artistic, mature",
        "Album URL": "https://example.com/15",
        "All Artists": "Pop Singer",
        "Album Type": "Album",
    },
    # Artist 5: 3 albums (metal - different style)
    {
        "Artist": "Metal Band",
        "Album": "Metal Debut",
        "Year": 2015,
        "Release Date": "June 6, 2015",
        "Genres": "Metal, Progressive Metal",
        "Critic Score": 68,
        "User Score": 72,
        "Avg Track Score": 70,
        "User Ratings": 250,
        "Critic Reviews": 7,
        "Tracks": 9,
        "Runtime (min)": 55.0,
        "Avg Track Runtime (min)": 6.11,
        "Label": "Metal Records",
        "Descriptors": "heavy, technical",
        "Album URL": "https://example.com/16",
        "All Artists": "Metal Band",
        "Album Type": "Album",
    },
    {
        "Artist": "Metal Band",
        "Album": "Metal Progression",
        "Year": 2018,
        "Release Date": "March 13, 2018",
        "Genres": "Metal, Progressive Metal",
        "Critic Score": 76,
        "User Score": 80,
        "Avg Track Score": 77,
        "User Ratings": 400,
        "Critic Reviews": 11,
        "Tracks": 8,
        "Runtime (min)": 60.0,
        "Avg Track Runtime (min)": 7.5,
        "Label": "Metal Records",
        "Descriptors": "ambitious, complex",
        "Album URL": "https://example.com/17",
        "All Artists": "Metal Band",
        "Album Type": "Album",
    },
    {
        "Artist": "Metal Band",
        "Album": "Metal Mastery",
        "Year": 2021,
        "Release Date": "October 31, 2021",
        "Genres": "Metal, Progressive Metal",
        "Critic Score": 82,
        "User Score": 85,
        "Avg Track Score": 83,
        "User Ratings": 600,
        "Critic Reviews": 15,
        "Tracks": 10,
        "Runtime (min)": 65.0,
        "Avg Track Runtime (min)": 6.5,
        "Label": "Major Metal",
        "Descriptors": "epic, masterful",
        "Album URL": "https://example.com/18",
        "All Artists": "Metal Band",
        "Album Type": "Album",
    },
    # Artist 6: 3 albums with EP to test album type variety
    {
        "Artist": "Indie Artist",
        "Album": "Indie Start",
        "Year": 2016,
        "Release Date": "September 10, 2016",
        "Genres": "Indie, Folk",
        "Critic Score": 70,
        "User Score": 73,
        "Avg Track Score": 71,
        "User Ratings": 180,
        "Critic Reviews": 5,
        "Tracks": 8,
        "Runtime (min)": 30.0,
        "Avg Track Runtime (min)": 3.75,
        "Label": "Indie Records",
        "Descriptors": "intimate, acoustic",
        "Album URL": "https://example.com/19",
        "All Artists": "Indie Artist",
        "Album Type": "Album",
    },
    {
        "Artist": "Indie Artist",
        "Album": "Indie EP",
        "Year": 2018,
        "Release Date": "February 14, 2018",
        "Genres": "Indie, Folk",
        "Critic Score": 72,
        "User Score": 75,
        "Avg Track Score": 73,
        "User Ratings": 150,
        "Critic Reviews": 4,
        "Tracks": 5,
        "Runtime (min)": 18.0,
        "Avg Track Runtime (min)": 3.6,
        "Label": "Indie Records",
        "Descriptors": "brief, beautiful",
        "Album URL": "https://example.com/20",
        "All Artists": "Indie Artist",
        "Album Type": "EP",
    },
    {
        "Artist": "Indie Artist",
        "Album": "Indie Full",
        "Year": 2020,
        "Release Date": "November 20, 2020",
        "Genres": "Indie, Indie Folk",
        "Critic Score": 80,
        "User Score": 83,
        "Avg Track Score": 80,
        "User Ratings": 350,
        "Critic Reviews": 10,
        "Tracks": 11,
        "Runtime (min)": 42.0,
        "Avg Track Runtime (min)": 3.82,
        "Label": "Major Indie",
        "Descriptors": "lush, expansive",
        "Album URL": "https://example.com/21",
        "All Artists": "Indie Artist",
        "Album Type": "Album",
    },
    # Collaboration album to test multi-artist handling
    {
        "Artist": "Test Artist One",
        "Album": "Collab Project",
        "Year": 2020,
        "Release Date": "May 1, 2020",
        "Genres": "Rock, Electronic",
        "Critic Score": 77,
        "User Score": 79,
        "Avg Track Score": 76,
        "User Ratings": 400,
        "Critic Reviews": 10,
        "Tracks": 9,
        "Runtime (min)": 38.0,
        "Avg Track Runtime (min)": 4.22,
        "Label": "Collab Records",
        "Descriptors": "fusion, innovative",
        "Album URL": "https://example.com/22",
        "All Artists": "Test Artist One | Electronic Producer",
        "Album Type": "Album",
    },
]


def create_minimal_dataset(tmp_path: Path) -> Path:
    """Create minimal test dataset in temporary directory.

    Creates a raw CSV file with valid album data for pipeline testing.
    The data includes:
    - 6 artists with 3+ albums each
    - Variety of genres, years, and album types
    - Valid numeric scores and ratings

    Args:
        tmp_path: Temporary directory path (pytest fixture).

    Returns:
        Path to the created CSV file.
    """
    # Create data/raw directory structure
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "all_albums_full.csv"

    # Write CSV with all columns in correct order
    fieldnames = [
        "Artist",
        "Album",
        "Year",
        "Release Date",
        "Genres",
        "Critic Score",
        "User Score",
        "Avg Track Score",
        "User Ratings",
        "Critic Reviews",
        "Tracks",
        "Runtime (min)",
        "Avg Track Runtime (min)",
        "Label",
        "Descriptors",
        "Album URL",
        "All Artists",
        "Album Type",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(MINIMAL_TEST_DATA)

    return csv_path


@pytest.fixture(scope="module")
def minimal_raw_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create minimal test CSV for E2E testing.

    Uses module scope to avoid recreating the file for each test.

    Yields:
        Path to the CSV file in a temporary directory.
    """
    tmp_path = tmp_path_factory.mktemp("e2e_data")
    return create_minimal_dataset(tmp_path)


@pytest.fixture(scope="module")
def e2e_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create temporary output directory for E2E tests.

    Uses module scope for efficiency across tests.

    Yields:
        Path to temporary output directory.
    """
    return tmp_path_factory.mktemp("e2e_outputs")


@pytest.fixture
def minimal_pipeline_config() -> PipelineConfig:
    """Create PipelineConfig optimized for fast testing.

    Configures the pipeline to:
    - Use fixed seed for reproducibility
    - Skip visualization stages
    - Run in non-strict mode

    Returns:
        PipelineConfig for fast test execution.
    """
    return PipelineConfig(
        seed=42,
        skip_existing=False,
        stages=["data", "splits", "features"],  # Skip MCMC stages
        dry_run=False,
        strict=False,
        verbose=False,
    )


@pytest.fixture
def dry_run_config() -> PipelineConfig:
    """Create PipelineConfig for dry run testing.

    Returns:
        PipelineConfig with dry_run=True.
    """
    return PipelineConfig(
        seed=42,
        dry_run=True,
        strict=False,
        verbose=False,
    )
