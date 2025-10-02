# Taxi Demand Forecasting Dashboard

A modular Streamlit application for time series forecasting of taxi demand.

## Project Structure

```
app/
├── __init__.py              # Package initialization
├── app.py                   # Main Streamlit application
├── config.py                # Configuration and constants
├── models.py                # Data models and types
├── database.py              # Database operations
├── forecaster_manager.py    # Forecaster lifecycle management
├── data_processor.py        # Data transformation
├── visualization.py         # Chart creation
└── ui_components.py         # Reusable UI elements

model_dev/
├── config.py                # Model configuration
└── forecast.py              # TimeSeriesForecaster class

results/                     # Output directory for results
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app/app.py
```

## Features

- **Modular Architecture**: Clean separation of concerns
- **Error Handling**: Comprehensive error handling and logging
- **Caching**: Efficient data caching for better performance
- **Visualization**: Interactive charts with Altair
- **Export**: Download forecast results as CSV
- **System Controls**: Restart forecaster and clear cache

## Configuration

Edit `app/config.py` to customize:
- Page settings
- Forecast parameters
- Database configuration
- Logging settings

## Development

Each module has a specific responsibility:

- **config.py**: All configuration constants
- **models.py**: Data classes and type definitions
- **database.py**: Database queries and connection management
- **forecaster_manager.py**: Forecaster initialization and execution
- **data_processor.py**: Data transformation and statistics
- **visualization.py**: Chart creation
- **ui_components.py**: Reusable Streamlit UI components
- **app.py**: Main application orchestration

## Testing

```bash
# Run unit tests (when implemented)
pytest tests/

# Run with verbose logging
streamlit run app/app.py --logger.level=debug
```

## License

[Your License]
"""