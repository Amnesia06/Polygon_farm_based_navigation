# Polygon Farm Based Navigation

This project implements an intelligent farm navigation system for autonomous agricultural rovers, 
focusing on efficient field coverage and optimal path planning in irregular farm shapes.

## Features 

- **Block-Based Farm Representation**
  - Converts irregular farm shapes into discrete rover-sized blocks
  - 100% sowable area guarantee
  - Support for various farm extensions (bottom, top, left, right)

- **Ultra-Efficient Navigation**
  - Zero-waste boustrophedon pattern
  - Optimal start position calculation
  - Intelligent path smoothing
  - No double-sowing guarantee
  - 100% coverage efficiency

- **Custom Exit Point Support**
  - User-selectable exit points
  - Border block identification
  - Efficient path adjustment for custom exits

- **Advanced Visualization**
  - Real-time sowing animation
  - Brown soil → Green sown visualization
  - Clear progress tracking
  - Performance metrics display

## System Requirements 

- Python 3.x
- Required packages:
  - numpy
  - matplotlib

## Installation 

1. Clone the repository:
```bash
git clone https://github.com/Amnesia06/Polygon_farm_based_navigation.git
cd Polygon_farm_based_navigation
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage 

### Main Program

Run the main program with:
```python
python polygon_navigation_enhanced.py
```

Choose from multiple options:
1. Ultra-Efficient Farm (System Chooses Best Exit)
2. Ultra-Efficient Farm (Custom Exit Point)
3. Navigation Efficiency Comparison
4. Quick Ultra-Efficient Demo
5. Test All Extension Patterns

### Testing

Run the test suite with:
```python
python test.py
```

Test options include:
1. Quick Demo
2. Run All Tests
3. Single Extension Test
4. Custom Extension Test
5. Rover Size Comparison

## Features in Detail 

### Block-Based Farm System
- Converts irregular farm shapes into discrete blocks
- Each block exactly matches rover dimensions
- Eliminates partial coverage issues
- Supports custom extension patterns

### Ultra-Efficient Navigation
- Intelligent start position selection
- Optimal boustrophedon pattern generation
- Path smoothing and optimization
- Complete coverage guarantee
- Minimal travel distance

### Performance Metrics
- Coverage efficiency
- Sowing precision
- Distance per block
- Total distance traveled
- Operation counts (sowing vs. navigation)


## Visualization Options 

1. **Static Analysis**
   - Farm structure visualization
   - Optimal path pattern
   - Performance metrics

2. **Dynamic Animation**
   - Real-time sowing visualization
   - Brown → Green transformation
   - Progress tracking
   - Coverage statistics

## Key Benefits

1. **Efficiency**
   - Zero wasted travel
   - Optimal path planning
   - Minimal distance coverage

2. **Reliability**
   - 100% coverage guarantee
   - No double-sowing
   - Complete area coverage

3. **Flexibility**
   - Support for irregular shapes
   - Custom exit points
   - Various farm extensions

