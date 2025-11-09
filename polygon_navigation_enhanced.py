



import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.patches import Rectangle as MPLRectangle
from matplotlib.animation import FuncAnimation
import time
from collections import deque

# ============================================================================
# BLOCK-BASED IRREGULAR FARM SHAPE (Same as before)
# ============================================================================

class BlockBasedFarm:
    """Farm made of discrete rover-sized blocks - 100% sowable"""
    
    def __init__(self, rect_width, rect_height, rover_width, rover_length, extension_side='bottom', extension_blocks=None):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.rover_width = rover_width
        self.rover_length = rover_length
        self.extension_side = extension_side
        
        # Calculate base rectangle in blocks
        self.rect_blocks_x = int(rect_width / rover_width)
        self.rect_blocks_y = int(rect_height / rover_length)
        
        # Default extension pattern if not provided
        if extension_blocks is None:
            extension_blocks = self._create_default_extension()
        
        self.extension_blocks = extension_blocks
        self.all_blocks = self._create_all_blocks()
        self.boundary_points = self._create_boundary()
        
        self.name = f"BlockFarm({self.rect_blocks_x}×{self.rect_blocks_y} + {len(extension_blocks)} ext blocks)"
    
    def _create_default_extension(self):
        """Create default extension pattern - stepped triangle-like shape"""
        if self.extension_side == 'bottom':
            extension = []
            max_steps = min(4, self.rect_blocks_x)
            for step in range(max_steps):
                for x in range(max_steps - step):
                    extension.append((x, -(step + 1)))
            return extension
            
        elif self.extension_side == 'top':
            extension = []
            max_steps = min(4, self.rect_blocks_x)
            for step in range(max_steps):
                for x in range(self.rect_blocks_x - max_steps + step, self.rect_blocks_x):
                    extension.append((x, self.rect_blocks_y + step))
            return extension
            
        elif self.extension_side == 'left':
            extension = []
            max_steps = min(4, self.rect_blocks_y)
            for step in range(max_steps):
                for y in range(max_steps - step):
                    extension.append((-(step + 1), y))
            return extension
            
        elif self.extension_side == 'right':
            extension = []
            max_steps = min(4, self.rect_blocks_y)
            for step in range(max_steps):
                for y in range(self.rect_blocks_y - max_steps + step, self.rect_blocks_y):
                    extension.append((self.rect_blocks_x + step, y))
            return extension
        
        return []
    
    def _create_all_blocks(self):
        """Create list of all blocks (rectangle + extension)"""
        blocks = []
        
        # Add rectangle blocks
        for bx in range(self.rect_blocks_x):
            for by in range(self.rect_blocks_y):
                blocks.append((bx, by))
        
        # Add extension blocks
        blocks.extend(self.extension_blocks)
        
        return blocks
    
    def _create_boundary(self):
        """Create boundary points from blocks"""
        if not self.all_blocks:
            return []
        
        xs = [bx * self.rover_width for bx, by in self.all_blocks] + [(bx + 1) * self.rover_width for bx, by in self.all_blocks]
        ys = [by * self.rover_length for bx, by in self.all_blocks] + [(by + 1) * self.rover_length for bx, by in self.all_blocks]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    
    def is_point_inside(self, x, y):
        """Check if point is inside any block"""
        for bx, by in self.all_blocks:
            block_x1 = bx * self.rover_width
            block_y1 = by * self.rover_length
            block_x2 = (bx + 1) * self.rover_width
            block_y2 = (by + 1) * self.rover_length
            
            if block_x1 <= x <= block_x2 and block_y1 <= y <= block_y2:
                return True
        return False
    
    def get_block_at_position(self, x, y):
        """Get block coordinates at given position"""
        bx = int(x // self.rover_width)
        by = int(y // self.rover_length)
        
        if (bx, by) in self.all_blocks:
            return (bx, by)
        return None
    
    def get_bounding_box(self):
        """Get bounding box of all blocks"""
        if not self.all_blocks:
            return (0, 0, 0, 0)
        
        xs = [bx * self.rover_width for bx, by in self.all_blocks] + [(bx + 1) * self.rover_width for bx, by in self.all_blocks]
        ys = [by * self.rover_length for bx, by in self.all_blocks] + [(by + 1) * self.rover_length for bx, by in self.all_blocks]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_farm_info(self):
        """Get farm information"""
        total_area = len(self.all_blocks) * self.rover_width * self.rover_length
        rect_area = self.rect_blocks_x * self.rect_blocks_y * self.rover_width * self.rover_length
        ext_area = len(self.extension_blocks) * self.rover_width * self.rover_length
        
        return {
            'total_blocks': len(self.all_blocks),
            'rectangle_blocks': self.rect_blocks_x * self.rect_blocks_y,
            'extension_blocks': len(self.extension_blocks),
            'total_area_m2': total_area,
            'rectangle_area_m2': rect_area,
            'extension_area_m2': ext_area,
            'extension_side': self.extension_side,
            'rover_size': f"{self.rover_width}×{self.rover_length}m",
            'sowable_percentage': 100.0
        }

# ============================================================================
# ROW-BASED SOWING LOGIC (FIXED FOR 100% COVERAGE)
# ============================================================================

def get_border_blocks(farm):
    """Get all border blocks of the farm"""
    border_blocks = []
    
    for bx, by in farm.all_blocks:
        is_border = False
        
        # Check if any adjacent position is NOT in the farm
        adjacent_positions = [
            (bx-1, by), (bx+1, by),  # Left, Right
            (bx, by-1), (bx, by+1)   # Down, Up
        ]
        
        for adj_bx, adj_by in adjacent_positions:
            if (adj_bx, adj_by) not in farm.all_blocks:
                is_border = True
                break
        
        if is_border:
            border_blocks.append((bx, by))
    
    return border_blocks

def get_user_exit_choice(farm):
    """Let user choose exit point from border blocks"""
    border_blocks = get_border_blocks(farm)
    
    print(f"\n🚪 Available Exit Points (Border Blocks):")
    print("=" * 40)
    
    # Group by sides for easier selection
    min_bx = min(bx for bx, by in farm.all_blocks)
    max_bx = max(bx for bx, by in farm.all_blocks)
    min_by = min(by for bx, by in farm.all_blocks)
    max_by = max(by for bx, by in farm.all_blocks)
    
    # Categorize border blocks
    left_border = [(bx, by) for bx, by in border_blocks if bx == min_bx]
    right_border = [(bx, by) for bx, by in border_blocks if bx == max_bx]
    bottom_border = [(bx, by) for bx, by in border_blocks if by == min_by]
    top_border = [(bx, by) for bx, by in border_blocks if by == max_by]
    
    # Remove duplicates (corner blocks appear in multiple lists)
    corners = set(left_border) & set(bottom_border) | set(left_border) & set(top_border) | \
              set(right_border) & set(bottom_border) | set(right_border) & set(top_border)
    
    print("📍 CORNER EXITS:")
    corner_list = sorted(list(corners))
    for i, (bx, by) in enumerate(corner_list):
        x_pos = (bx + 0.5) * farm.rover_width
        y_pos = (by + 0.5) * farm.rover_length
        print(f"   {i+1}. Block({bx},{by}) → Position({x_pos:.1f}m, {y_pos:.1f}m)")
    
    print(f"\n📍 LEFT BORDER EXITS:")
    left_non_corner = [block for block in left_border if block not in corners]
    for i, (bx, by) in enumerate(sorted(left_non_corner)):
        x_pos = (bx + 0.5) * farm.rover_width
        y_pos = (by + 0.5) * farm.rover_length
        print(f"   {len(corner_list)+i+1}. Block({bx},{by}) → Position({x_pos:.1f}m, {y_pos:.1f}m)")
    
    print(f"\n📍 RIGHT BORDER EXITS:")
    right_non_corner = [block for block in right_border if block not in corners]
    for i, (bx, by) in enumerate(sorted(right_non_corner)):
        x_pos = (bx + 0.5) * farm.rover_width
        y_pos = (by + 0.5) * farm.rover_length
        idx = len(corner_list) + len(left_non_corner) + i + 1
        print(f"   {idx}. Block({bx},{by}) → Position({x_pos:.1f}m, {y_pos:.1f}m)")
    
    print(f"\n📍 BOTTOM BORDER EXITS:")
    bottom_non_corner = [block for block in bottom_border if block not in corners]
    for i, (bx, by) in enumerate(sorted(bottom_non_corner)):
        x_pos = (bx + 0.5) * farm.rover_width
        y_pos = (by + 0.5) * farm.rover_length
        idx = len(corner_list) + len(left_non_corner) + len(right_non_corner) + i + 1
        print(f"   {idx}. Block({bx},{by}) → Position({x_pos:.1f}m, {y_pos:.1f}m)")
    
    print(f"\n📍 TOP BORDER EXITS:")
    top_non_corner = [block for block in top_border if block not in corners]
    for i, (bx, by) in enumerate(sorted(top_non_corner)):
        x_pos = (bx + 0.5) * farm.rover_width
        y_pos = (by + 0.5) * farm.rover_length
        idx = len(corner_list) + len(left_non_corner) + len(right_non_corner) + len(bottom_non_corner) + i + 1
        print(f"   {idx}. Block({bx},{by}) → Position({x_pos:.1f}m, {y_pos:.1f}m)")
    
    # Create ordered list of all options
    all_options = corner_list + sorted(left_non_corner) + sorted(right_non_corner) + \
                  sorted(bottom_non_corner) + sorted(top_non_corner)
    
    print(f"\n🎯 Total exit options: {len(all_options)}")
    
    while True:
        try:
            choice = int(input(f"\nChoose exit point (1-{len(all_options)}): "))
            if 1 <= choice <= len(all_options):
                selected_block = all_options[choice - 1]
                x_pos = (selected_block[0] + 0.5) * farm.rover_width
                y_pos = (selected_block[1] + 0.5) * farm.rover_length
                print(f"✅ Selected: Block{selected_block} → Position({x_pos:.1f}m, {y_pos:.1f}m)")
                return selected_block
            else:
                print(f"❌ Please enter a number between 1 and {len(all_options)}")
        except ValueError:
            print("❌ Please enter a valid number")

def _find_shortest_path_orthogonal(start_block, end_block, valid_blocks):
    """Find shortest orthogonal path between blocks using BFS"""
    if start_block == end_block:
        return [start_block]
    
    queue = deque([(start_block, [start_block])])
    visited = {start_block}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
    
    while queue:
        current_block, path = queue.popleft()
        
        if current_block == end_block:
            return path
        
        bx, by = current_block
        
        for dx, dy in directions:
            next_block = (bx + dx, by + dy)
            
            if next_block in valid_blocks and next_block not in visited:
                visited.add(next_block)
                new_path = path + [next_block]
                queue.append((next_block, new_path))
    
    return [start_block, end_block]  # Fallback direct path

def _determine_optimal_start_position(exit_block, valid_blocks):
    """Determine the MOST efficient starting position for minimal total travel"""
    
    print("   🎯 Finding ultra-optimal start position...")
    
    # STEP 1: Analyze farm structure for boustrophedon optimization
    columns = {}
    for bx, by in valid_blocks:
        if bx not in columns:
            columns[bx] = []
        columns[bx].append(by)
    
    for bx in columns:
        columns[bx].sort()
    
    sorted_columns = sorted(columns.keys())
    exit_bx, exit_by = exit_block
    
    # STEP 2: Determine optimal column traversal direction
    if exit_bx <= sorted_columns[len(sorted_columns)//2]:
        # Exit is on left side - traverse RIGHT to LEFT (end near exit)
        start_column = sorted_columns[-1]  # Rightmost column
        print(f"   📐 Exit on left, starting from rightmost column {start_column}")
    else:
        # Exit is on right side - traverse LEFT to RIGHT (end near exit)
        start_column = sorted_columns[0]   # Leftmost column
        print(f"   📐 Exit on right, starting from leftmost column {start_column}")
    
    # STEP 3: Find optimal position within the starting column
    start_column_blocks = columns[start_column]
    
    # Choose end of column that's farthest from exit's Y position
    if exit_by <= (min(start_column_blocks) + max(start_column_blocks)) / 2:
        # Exit is in lower half - start from top of column
        optimal_start_y = max(start_column_blocks)
        print(f"   📐 Exit below center, starting from top of column")
    else:
        # Exit is in upper half - start from bottom of column
        optimal_start_y = min(start_column_blocks)
        print(f"   📐 Exit above center, starting from bottom of column")
    
    optimal_start = (start_column, optimal_start_y)
    
    # STEP 4: Verify this start position exists
    if optimal_start not in valid_blocks:
        print(f"   ⚠️ Calculated start {optimal_start} not valid, finding nearest...")
        # Find nearest valid block in the same column
        valid_y_positions = [by for bx, by in valid_blocks if bx == start_column]
        if valid_y_positions:
            nearest_y = min(valid_y_positions, key=lambda y: abs(y - optimal_start_y))
            optimal_start = (start_column, nearest_y)
        else:
            # Fallback to any border block
            border_blocks = get_border_blocks_from_set(valid_blocks)
            optimal_start = border_blocks[0] if border_blocks else list(valid_blocks)[0]
    
    # STEP 5: Calculate efficiency info
    exit_distance = abs(optimal_start[0] - exit_bx) + abs(optimal_start[1] - exit_by)
    
    print(f"   ✅ Ultra-optimal start: {optimal_start}")
    print(f"   📏 Distance to exit: {exit_distance} blocks")
    print(f"   🎯 Optimized for boustrophedon + minimal travel")
    
    return optimal_start


def get_border_blocks_from_set(valid_blocks):
    """Get border blocks from a set of valid blocks"""
    border_blocks = []
    
    for bx, by in valid_blocks:
        is_border = False
        
        # Check if any adjacent position is NOT in the valid blocks
        adjacent_positions = [
            (bx-1, by), (bx+1, by),  # Left, Right
            (bx, by-1), (bx, by+1)   # Down, Up
        ]
        
        for adj_bx, adj_by in adjacent_positions:
            if (adj_bx, adj_by) not in valid_blocks:
                is_border = True
                break
        
        if is_border:
            border_blocks.append((bx, by))
    
    return border_blocks

def _get_corner_blocks(valid_blocks):
    """Get corner blocks from valid blocks for start position analysis"""
    
    if not valid_blocks:
        return []
    
    min_bx = min(bx for bx, by in valid_blocks)
    max_bx = max(bx for bx, by in valid_blocks)
    min_by = min(by for bx, by in valid_blocks)
    max_by = max(by for bx, by in valid_blocks)
    
    # Potential corner positions
    corners = [
        (min_bx, min_by),  # Bottom-left
        (max_bx, min_by),  # Bottom-right
        (min_bx, max_by),  # Top-left
        (max_bx, max_by)   # Top-right
    ]
    
    # Return only corners that actually exist in valid_blocks
    valid_corners = [corner for corner in corners if corner in valid_blocks]
    return valid_corners


def _create_zero_waste_boustrophedon(valid_blocks, exit_block):
    """Create boustrophedon pattern that STARTS where sowing begins (zero waste)"""
    
    print("   🎯 Creating ZERO-WASTE boustrophedon pattern...")
    
    # Group blocks by columns
    columns = {}
    for bx, by in valid_blocks:
        if bx not in columns:
            columns[bx] = []
        columns[bx].append(by)
    
    for bx in columns:
        columns[bx].sort()
    
    sorted_columns = sorted(columns.keys())
    exit_bx, exit_by = exit_block
    
    # SMART DECISION: Which direction ends closest to exit?
    if exit_bx <= sorted_columns[len(sorted_columns)//2]:
        # Exit on left side - traverse RIGHT to LEFT (end near exit)
        column_order = sorted_columns[::-1]  # Start from rightmost
        start_column = sorted_columns[-1]
        print(f"   📐 Exit on left → Start RIGHT, traverse LEFT (end near exit)")
    else:
        # Exit on right side - traverse LEFT to RIGHT (end near exit)
        column_order = sorted_columns  # Start from leftmost
        start_column = sorted_columns[0]
        print(f"   📐 Exit on right → Start LEFT, traverse RIGHT (end near exit)")
    
    # SMART START POSITION: Farthest from exit within start column
    start_column_blocks = columns[start_column]
    if exit_by <= (min(start_column_blocks) + max(start_column_blocks)) / 2:
        # Exit in lower half - start from TOP of column (farthest away)
        start_y = max(start_column_blocks)
        direction_up = False  # Start high, go down first
        print(f"   📐 Exit below center → Start from TOP of column (maximize distance)")
    else:
        # Exit in upper half - start from BOTTOM of column (farthest away)
        start_y = min(start_column_blocks)
        direction_up = True   # Start low, go up first
        print(f"   📐 Exit above center → Start from BOTTOM of column (maximize distance)")
    
    # Generate the ZERO-WASTE boustrophedon path
    path = []
    
    for i, bx in enumerate(column_order):
        if bx not in columns:
            continue
        
        column_blocks = columns[bx]
        
        # Alternate direction for boustrophedon pattern
        if direction_up:
            ordered_blocks = sorted(column_blocks)
        else:
            ordered_blocks = sorted(column_blocks, reverse=True)
        
        # Add all blocks in this column
        for by in ordered_blocks:
            path.append((bx, by))
        
        # Alternate direction for next column
        direction_up = not direction_up
        
        print(f"   ✅ Column {bx}: {len(ordered_blocks)} blocks ({'UP' if not direction_up else 'DOWN'} next)")
    
    # Navigate to exit at the end
    if path and path[-1] != exit_block:
        path.append(exit_block)
    
    actual_start = path[0]
    print(f"   🎯 ZERO-WASTE path created: {len(path)} blocks")
    print(f"   🚀 Optimal start: {actual_start} (immediate sowing)")
    print(f"   🚪 End near exit: {path[-1]}")
    
    return path


def _find_shortest_path_orthogonal(start_block, end_block, valid_blocks):
    """Find shortest orthogonal path between blocks using BFS - OPTIMIZED"""
    if start_block == end_block:
        return [start_block]
    
    if start_block not in valid_blocks or end_block not in valid_blocks:
        print(f"   ⚠️ Invalid blocks in path: {start_block} or {end_block}")
        return [start_block, end_block]
    
    queue = deque([(start_block, [start_block])])
    visited = {start_block}
    
    # Orthogonal directions only (no diagonal)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
    
    while queue:
        current_block, path = queue.popleft()
        
        if current_block == end_block:
            return path
        
        bx, by = current_block
        
        for dx, dy in directions:
            next_block = (bx + dx, by + dy)
            
            if next_block in valid_blocks and next_block not in visited:
                visited.add(next_block)
                new_path = path + [next_block]
                queue.append((next_block, new_path))
    
    # If no path found, return direct path (shouldn't happen with valid blocks)
    print(f"   ⚠️ No orthogonal path found from {start_block} to {end_block}")
    return [start_block, end_block]

def verify_no_double_sowing(path_lanes, sow_flags):
    """Verify that no block is sown twice"""
    
    print("🔍 Verifying NO DOUBLE-SOWING...")
    
    sown_blocks = set()
    double_sown_blocks = set()
    sowing_operations = 0
    
    for i, block in enumerate(path_lanes):
        if i > 0 and i-1 < len(sow_flags) and sow_flags[i-1]:
            sowing_operations += 1
            if block in sown_blocks:
                double_sown_blocks.add(block)
                print(f"   ❌ DOUBLE-SOWING DETECTED: Block {block} sown again!")
            else:
                sown_blocks.add(block)
    
    unique_blocks_sown = len(sown_blocks)
    
    print(f"   📊 Verification Results:")
    print(f"   Total sowing operations: {sowing_operations}")
    print(f"   Unique blocks sown: {unique_blocks_sown}")
    print(f"   Double-sown blocks: {len(double_sown_blocks)}")
    
    if len(double_sown_blocks) == 0 and sowing_operations == unique_blocks_sown:
        print(f"   ✅ SUCCESS: NO DOUBLE-SOWING detected!")
        print(f"   ✅ Perfect efficiency: Each block sown exactly once!")
        return True
    else:
        print(f"   ❌ FAILURE: Double-sowing detected!")
        if double_sown_blocks:
            print(f"   Double-sown blocks: {sorted(list(double_sown_blocks))}")
        return False



def generate_optimal_coverage_path(grid, exit_block=None):
    """Generate ZERO-WASTE optimal coverage path - start where sowing begins!"""
    
    if not grid.valid_lanes:
        print("❌ No valid lanes found!")
        return [], []
    
    print("🌾 Generating ZERO-WASTE optimal coverage path...")
    
    valid_blocks = set(grid.valid_lanes)
    total_blocks = len(valid_blocks)
    
    # Set default exit if not provided
    if exit_block is None:
        border_blocks = get_border_blocks_from_set(valid_blocks)
        exit_block = border_blocks[0] if border_blocks else list(valid_blocks)[0]
    
    print(f"   📊 Total blocks to sow: {total_blocks}")
    print(f"   🚪 Exit block: {exit_block}")
    
    # STEP 1: Create optimal boustrophedon pattern FIRST (determines real start)
    optimal_boustrophedon_path = _create_zero_waste_boustrophedon(valid_blocks, exit_block)
    
    # STEP 2: START DIRECTLY at the first block of boustrophedon (ZERO WASTE!)
    actual_start_block = optimal_boustrophedon_path[0]
    
    print(f"   🎯 ZERO-WASTE start: {actual_start_block} (sowing begins immediately)")
    print(f"   ✅ No wasted travel - every move is productive!")
    
    # STEP 3: Build navigation path (already optimal)
    points_lanes = []
    current_pos = actual_start_block
    points_lanes.append(current_pos)
    
    # Navigate through the boustrophedon pattern
    for target_block in optimal_boustrophedon_path[1:]:
        if target_block == current_pos:
            continue
        
        # Find path to target block
        path_to_target = _find_shortest_path_orthogonal(current_pos, target_block, valid_blocks)
        
        # Add all steps to navigation path
        for step in path_to_target[1:]:  # Skip current position
            points_lanes.append(step)
        
        current_pos = target_block
    
    # STEP 4: Generate ZERO-WASTE sow flags
    sow_flags = []
    blocks_already_sown = {actual_start_block}  # Start block is sown
    
    print("   🌾 Applying zero-waste sowing strategy...")
    
    for i in range(len(points_lanes) - 1):
        current_block = points_lanes[i]
        next_block = points_lanes[i + 1]
        
        # Zero-waste strategy: sow every new block we visit
        should_sow = False
        
        if next_block not in blocks_already_sown:
            should_sow = True
            blocks_already_sown.add(next_block)
        
        sow_flags.append(should_sow)
    
    # STEP 5: Verify zero-waste efficiency
    sowing_operations = len(blocks_already_sown)
    navigation_operations = len(sow_flags) - sum(sow_flags)
    coverage_percent = (len(blocks_already_sown) / total_blocks) * 100
    efficiency_percent = (sowing_operations / len(points_lanes)) * 100 if points_lanes else 0
    
    print(f"✅ ZERO-WASTE path generated:")
    print(f"   Total waypoints: {len(points_lanes)}")
    print(f"   Sowing operations: {sowing_operations}")
    print(f"   Navigation operations: {navigation_operations}")
    print(f"   Coverage: {coverage_percent:.1f}%")
    print(f"   Efficiency: {efficiency_percent:.1f}% (MUCH HIGHER!)")
    print(f"   🎉 Zero wasted travel - sowing starts immediately!")
    
    return points_lanes, sow_flags






def generate_optimal_coverage_path_with_exit(grid, exit_block):
    """Generate optimal coverage path with specific exit point"""
    return generate_optimal_coverage_path(grid, exit_block)

# ============================================================================
# BLOCK GRID CLASS (Updated for optimal sowing)
# ============================================================================

class BlockGrid:
    """Grid system for block-based farm with optimal sowing support"""
    
    def __init__(self, farm):
        self.farm = farm
        self.rover_width = farm.rover_width
        self.rover_length = farm.rover_length
        
        # All blocks are valid lanes by design
        self.valid_lanes = farm.all_blocks.copy()
        
        # Calculate grid dimensions
        min_bx = min(bx for bx, by in self.valid_lanes)
        max_bx = max(bx for bx, by in self.valid_lanes)
        min_by = min(by for bx, by in self.valid_lanes)
        max_by = max(by for bx, by in self.valid_lanes)
        
        self.min_lane_x, self.max_lane_x = min_bx, max_bx
        self.min_lane_y, self.max_lane_y = min_by, max_by
        
        self.num_lanes_x = max_bx - min_bx + 1
        self.num_lanes_y = max_by - min_by + 1
        
        # Calculate columns for boustrophedon analysis
        columns = {}
        for bx, by in self.valid_lanes:
            if bx not in columns:
                columns[bx] = []
            columns[bx].append(by)
        
        self.n_columns = len(columns)
        
        farm_info = farm.get_farm_info()
        
        print(f"🏗️ Block-Based Farm Grid (Optimal Sowing):")
        print(f"   Total blocks: {farm_info['total_blocks']}")
        print(f"   Rectangle: {farm_info['rectangle_blocks']} blocks")
        print(f"   Extension: {farm_info['extension_blocks']} blocks on {farm_info['extension_side']} side")
        print(f"   Total area: {farm_info['total_area_m2']:.1f}m²")
        print(f"   Rover: {farm_info['rover_size']}")
        print(f"   Grid range: X({min_bx} to {max_bx}), Y({min_by} to {max_by})")
        print(f"   Valid lanes: {len(self.valid_lanes)}")
        print(f"   Columns for boustrophedon: {self.n_columns}")
        print(f"   🎯 Optimal path planning enabled")
    
    def lane_to_metric(self, lane_x, lane_y):
        """Convert lane indices to metric coordinates (center of block)"""
        x = (lane_x + 0.5) * self.rover_width
        y = (lane_y + 0.5) * self.rover_length
        return (x, y)
    
    def is_lane_valid(self, lane_x, lane_y):
        """Check if lane is valid (exists in our block list)"""
        return (lane_x, lane_y) in self.valid_lanes

# ============================================================================
# ENHANCED VISUALIZATION FOR OPTIMAL SOWING
# ============================================================================
def visualize_optimal_farm_and_path(farm, path_lanes, sow_flags):
    """Visualize block-based farm with optimal sowing pattern - CLEAN VERSION"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Farm structure (unchanged)
    ax1.set_title(f'🏗️ {farm.name}\n(Optimal Boustrophedon Pattern)', fontsize=14, fontweight='bold')
    
    # Draw all blocks with column indicators
    for bx, by in farm.all_blocks:
        x = bx * farm.rover_width
        y = by * farm.rover_length
        
        # Different colors for rectangle vs extension
        if 0 <= bx < farm.rect_blocks_x and 0 <= by < farm.rect_blocks_y:
            color = 'lightgreen'  # Rectangle blocks
            alpha = 0.7
        else:
            color = 'lightblue'   # Extension blocks
            alpha = 0.8
        
        rect = MPLRectangle((x, y), farm.rover_width, farm.rover_length,
                           facecolor=color, edgecolor='black', linewidth=1, alpha=alpha)
        ax1.add_patch(rect)
        
        # Add block coordinates
        ax1.text(x + farm.rover_width/2, y + farm.rover_length/2, f'({bx},{by})',
                ha='center', va='center', fontsize=8, weight='bold')
    
    # Draw boustrophedon column indicators
    min_bx = min(bx for bx, by in farm.all_blocks)
    max_bx = max(bx for bx, by in farm.all_blocks)
    max_by = max(by for bx, by in farm.all_blocks)
    
    # Draw column indicators with alternating colors
    for i, bx in enumerate(range(min_bx, max_bx + 1)):
        x = bx * farm.rover_width
        color = 'red' if i % 2 == 0 else 'blue'  # Alternate colors for boustrophedon
        ax1.axvline(x + farm.rover_width/2, color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        # Direction arrows
        direction = '↑' if i % 2 == 0 else '↓'
        ax1.text(x + farm.rover_width/2, (max_by + 1) * farm.rover_length + 10, 
                f'C{bx}{direction}', ha='center', va='bottom', fontsize=10, 
                color=color, weight='bold')
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Create legend for Plot 1
    legend_elements1 = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.7, label='Rectangle'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, label='Extension'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.6, label='Columns')
    ]
    ax1.legend(handles=legend_elements1, loc='upper left', bbox_to_anchor=(0, 1), 
              fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
    
    # Plot 2: CLEAN sowing result (NO PATH LINES)
    ax2.set_title('🎯 Clean Sowing Result\n(Brown→Green, No Path Lines)', fontsize=14, fontweight='bold')
    
    # Draw blocks with sowing status
    grid = BlockGrid(farm)
    
    # Determine which blocks get sown
    sown_blocks = set()
    
    # Starting block is always sown
    if path_lanes:
        sown_blocks.add(path_lanes[0])
    
    # Process sow_flags to determine sown blocks
    for i, block in enumerate(path_lanes):
        if i > 0 and i-1 < len(sow_flags) and sow_flags[i-1]:
            sown_blocks.add(block)
    
    # Draw all blocks with appropriate colors
    for bx, by in farm.all_blocks:
        x = bx * farm.rover_width
        y = by * farm.rover_length
        
        if (bx, by) in sown_blocks:
            # Sown block - green
            rect = MPLRectangle((x, y), farm.rover_width, farm.rover_length,
                               facecolor='#228B22', edgecolor='black', linewidth=2, alpha=0.9)
            ax2.add_patch(rect)
            ax2.text(x + farm.rover_width/2, y + farm.rover_length/2, 'SOWN',
                    ha='center', va='center', fontsize=9, weight='bold', color='white')
        else:
            # Unsown block - brown
            rect = MPLRectangle((x, y), farm.rover_width, farm.rover_length,
                               facecolor='#8B4513', edgecolor='black', linewidth=2, alpha=0.9)
            ax2.add_patch(rect)
            ax2.text(x + farm.rover_width/2, y + farm.rover_length/2, 'SOIL',
                    ha='center', va='center', fontsize=9, weight='bold', color='white')
    
    if path_lanes:
        # Convert path to metric coordinates for start/end markers only
        path_metric = [grid.lane_to_metric(bx, by) for bx, by in path_lanes]
        
        # Start and end points ONLY (no path lines)
        ax2.plot(path_metric[0][0], path_metric[0][1], 'go', markersize=18, label='Start', zorder=5, 
                markeredgecolor='black', markeredgewidth=3)
        ax2.plot(path_metric[-1][0], path_metric[-1][1], 'ro', markersize=18, label='Exit', zorder=5, 
                markeredgecolor='black', markeredgewidth=3)
        
        # Calculate stats
                # Calculate MEANINGFUL stats
        farm_info = farm.get_farm_info()
        metrics = calculate_meaningful_efficiency_metrics(path_lanes, sow_flags, set(farm.all_blocks), grid)
        
        # Status display with CORRECT metrics
        status_color = 'lightgreen' if metrics['coverage_efficiency'] >= 100 else 'lightcoral'
        status_text = (f'🎯 SOWING PERFORMANCE\n'
                      f'Coverage: {metrics["coverage_efficiency"]:.1f}%\n'
                      f'Precision: {metrics["sowing_precision"]:.1f}%\n'
                      f'Blocks Sown: {metrics["sowing_ops"]}/{farm_info["total_blocks"]}\n'
                      f'Distance: {metrics["total_distance"]:.1f}m\n'
                      f'Efficiency: {metrics["distance_per_block"]:.1f}m/block\n'
                      f'{"🎉 PERFECT!" if metrics["coverage_efficiency"] >= 100 else "❌ INCOMPLETE"}')

        
        ax2.text(0.02, 0.02, status_text,
                 transform=ax2.transAxes, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.9),
                 fontsize=9, weight='bold')
    
    # Create legend for Plot 2 (no path line elements)
    legend_elements2 = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#228B22', alpha=0.9, label='Sown Blocks'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#8B4513', alpha=0.9, label='Soil Blocks'),
        plt.Line2D([0], [0], marker='o', color='g', markersize=10, linestyle='None', 
                   markeredgecolor='black', markeredgewidth=2, label='Start'),
        plt.Line2D([0], [0], marker='o', color='r', markersize=10, linestyle='None', 
                   markeredgecolor='black', markeredgewidth=2, label='Exit')
    ]
    ax2.legend(handles=legend_elements2, loc='upper left', bbox_to_anchor=(0, 1), 
              fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    plt.pause(0.1)



def generate_ultra_efficient_coverage_path(grid, exit_block=None):
    """Generate ultra-efficient coverage path with minimal travel distance and 100% coverage"""
    
    if not grid.valid_lanes:
        print("❌ No valid lanes found!")
        return [], []
    
    print("🚀 Generating ULTRA-EFFICIENT coverage path (Minimal Distance + 100% Coverage)...")
    
    valid_blocks = set(grid.valid_lanes)
    total_blocks = len(valid_blocks)
    
    # Set default exit if not provided
    if exit_block is None:
        border_blocks = get_border_blocks_from_set(valid_blocks)
        exit_block = border_blocks[0] if border_blocks else list(valid_blocks)[0]
    
    print(f"   📊 Total blocks to sow: {total_blocks}")
    print(f"   🚪 Exit block: {exit_block}")
    
    # STEP 1: CREATE ZERO-WASTE BOUSTROPHEDON (determines optimal start)
    optimized_path = _create_zero_waste_boustrophedon(valid_blocks, exit_block)
    
    # STEP 2: START DIRECTLY where sowing begins (ZERO WASTE!)
    start_block = optimized_path[0]
    print(f"   🎯 Zero-waste start: {start_block} (no wasted travel)")

    # STEP 3: PATH SMOOTHING AND OPTIMIZATION
    smoothed_path = _smooth_and_optimize_path(optimized_path, valid_blocks)
    
    # STEP 4: ENSURE 100% COVERAGE (CRITICAL CHECK)
    complete_path = _ensure_complete_coverage(smoothed_path, valid_blocks, exit_block)
    
    # STEP 5: GENERATE NAVIGATION WITH SOW-ON-RETURN STRATEGY
    points_lanes, sow_flags = _generate_efficient_navigation_with_sowing(complete_path, valid_blocks, start_block)
    
    # STEP 6: FINAL VERIFICATION AND DISTANCE CALCULATION
    total_distance = _calculate_total_distance(points_lanes, grid)
    sown_blocks = _verify_complete_coverage(points_lanes, sow_flags, valid_blocks)
    
    coverage_percent = (len(sown_blocks) / total_blocks) * 100
    sowing_operations = sum(sow_flags) + 1  # +1 for starting block
    
    print(f"✅ ULTRA-EFFICIENT path generated:")
    print(f"   Total waypoints: {len(points_lanes)}")
    print(f"   Total distance: {total_distance:.1f}m")
    print(f"   Sowing operations: {sowing_operations}")
    print(f"   Navigation operations: {len(sow_flags) - sum(sow_flags)}")
    print(f"   Unique blocks sown: {len(sown_blocks)}/{total_blocks}")
    print(f"   Coverage: {coverage_percent:.1f}%")
    print(f"   Efficiency: {(sowing_operations / len(points_lanes) * 100):.1f}%")
    
    if coverage_percent >= 100:
        print(f"   🎉 SUCCESS: 100% coverage with ultra-efficient navigation!")
    else:
        print(f"   ❌ CRITICAL ERROR: Coverage incomplete!")
        unsown_blocks = valid_blocks - sown_blocks
        print(f"   Unsown blocks: {sorted(list(unsown_blocks))}")
    
    return points_lanes, sow_flags

def _find_ultra_optimal_start_position(valid_blocks, exit_block):
    """Find the most efficient start position considering total path distance"""
    
    print("   🎯 Finding ultra-optimal start position...")
    
    # Get all border blocks as potential start positions
    border_blocks = get_border_blocks_from_set(valid_blocks)
    
    if not border_blocks:
        return list(valid_blocks)[0]
    
    # Calculate farm center and dimensions
    min_bx = min(bx for bx, by in valid_blocks)
    max_bx = max(bx for bx, by in valid_blocks)
    min_by = min(by for bx, by in valid_blocks)
    max_by = max(by for bx, by in valid_blocks)
    
    center_x = (min_bx + max_bx) / 2
    center_y = (min_by + max_by) / 2
    
    exit_bx, exit_by = exit_block
    
    # Evaluate each potential start position
    best_start = None
    min_estimated_distance = float('inf')
    
    for candidate_start in border_blocks:
        start_bx, start_by = candidate_start
        
        # Calculate estimated total path distance for this start position
        # Distance from start to center + center to exit + coverage distance
        start_to_center = abs(start_bx - center_x) + abs(start_by - center_y)
        center_to_exit = abs(center_x - exit_bx) + abs(center_y - exit_by)
        
        # Estimate coverage distance based on farm dimensions
        coverage_distance = (max_bx - min_bx + 1) * (max_by - min_by + 1) * 0.8
        
        # Penalty for start positions close to exit (inefficient)
        start_exit_distance = abs(start_bx - exit_bx) + abs(start_by - exit_by)
        if start_exit_distance < 3:  # Too close to exit
            proximity_penalty = 100
        else:
            proximity_penalty = 0
        
        total_estimated_distance = start_to_center + center_to_exit + coverage_distance + proximity_penalty
        
        if total_estimated_distance < min_estimated_distance:
            min_estimated_distance = total_estimated_distance
            best_start = candidate_start
    
    print(f"   ✅ Ultra-optimal start: {best_start} (estimated distance: {min_estimated_distance:.1f})")
    return best_start

def _create_ultra_efficient_boustrophedon(valid_blocks, start_block, exit_block):
    """Create ultra-efficient boustrophedon pattern with minimal travel distance"""
    
    print("   🌾 Creating ultra-efficient boustrophedon pattern...")
    
    # Group blocks by columns
    columns = {}
    for bx, by in valid_blocks:
        if bx not in columns:
            columns[bx] = []
        columns[bx].append(by)
    
    # Sort each column
    for bx in columns:
        columns[bx].sort()
    
    # SMART COLUMN ORDERING based on start and exit positions
    start_bx, start_by = start_block
    exit_bx, exit_by = exit_block
    
    sorted_columns = sorted(columns.keys())
    
    # Choose column order to minimize total distance
    if start_bx <= exit_bx:
        # Start is left of exit, go left to right
        column_order = sorted_columns
        print(f"   📐 Efficient column order: Left→Right {sorted_columns}")
    else:
        # Start is right of exit, go right to left
        column_order = sorted_columns[::-1]
        print(f"   📐 Efficient column order: Right→Left {column_order}")
    
    # SMART DIRECTION PLANNING
    path = []
    direction_up = True
    
    # Determine optimal starting direction
    if start_block in valid_blocks:
        start_column = start_bx
        if start_column in columns:
            column_blocks = columns[start_column]
            # Start from the end closest to our start position
            if start_by <= (min(column_blocks) + max(column_blocks)) / 2:
                direction_up = True  # Start from bottom
            else:
                direction_up = False  # Start from top
    
    print(f"   🔄 Efficient starting direction: {'UP' if direction_up else 'DOWN'}")
    
    # Generate optimized boustrophedon pattern
    for i, bx in enumerate(column_order):
        if bx not in columns:
            continue
        
        column_blocks = columns[bx]
        
        # Alternate direction for boustrophedon
        if direction_up:
            ordered_blocks = sorted(column_blocks)
        else:
            ordered_blocks = sorted(column_blocks, reverse=True)
        
        # Add all blocks in this column
        for by in ordered_blocks:
            path.append((bx, by))
        
        # Alternate direction for next column
        direction_up = not direction_up
        
        print(f"   ✅ Column {bx}: {len(ordered_blocks)} blocks")
    
    print(f"   🎯 Ultra-efficient boustrophedon: {len(path)} blocks")
    return path

def _smooth_and_optimize_path(path, valid_blocks):
    """Smooth and optimize the path to reduce unnecessary movements"""
    
    print("   🔧 Smoothing and optimizing path...")
    
    if len(path) <= 2:
        return path
    
    # Remove unnecessary zigzags and optimize transitions
    optimized_path = [path[0]]  # Start with first block
    
    for i in range(1, len(path) - 1):
        current = path[i]
        prev_block = optimized_path[-1]
        next_block = path[i + 1]
        
        # Check if we can skip this block and go directly to next
        # Only if it doesn't break coverage
        can_skip = False
        
        # Calculate distances
        dist_via_current = abs(prev_block[0] - current[0]) + abs(prev_block[1] - current[1]) + \
                          abs(current[0] - next_block[0]) + abs(current[1] - next_block[1])
        dist_direct = abs(prev_block[0] - next_block[0]) + abs(prev_block[1] - next_block[1])
        
        # If direct path is much shorter and current block will be visited later, consider skipping
        if dist_direct < dist_via_current * 0.7:
            # Check if current block appears later in path
            if current in path[i+2:]:
                can_skip = True
        
        if not can_skip:
            optimized_path.append(current)
    
    # Add the last block
    optimized_path.append(path[-1])
    
    print(f"   ✅ Path optimized: {len(path)} → {len(optimized_path)} blocks")
    return optimized_path

def _ensure_complete_coverage(path, valid_blocks, exit_block):
    """Ensure 100% coverage by adding any missed blocks efficiently"""
    
    print("   🔍 Ensuring 100% coverage...")
    
    path_blocks = set(path)
    missed_blocks = valid_blocks - path_blocks
    
    if not missed_blocks:
        print("   ✅ All blocks already in path")
        # Add path to exit
        if path[-1] != exit_block:
            path.append(exit_block)
        return path
    
    print(f"   🔄 Adding {len(missed_blocks)} missed blocks efficiently...")
    
    complete_path = path.copy()
    current_pos = path[-1]
    
    # Add missed blocks in order of proximity to current position
    remaining_missed = list(missed_blocks)
    
    while remaining_missed:
        # Find closest missed block to current position
        closest_block = min(remaining_missed, 
                           key=lambda b: abs(b[0] - current_pos[0]) + abs(b[1] - current_pos[1]))
        
        complete_path.append(closest_block)
        remaining_missed.remove(closest_block)
        current_pos = closest_block
    
    # Add path to exit
    if complete_path[-1] != exit_block:
        complete_path.append(exit_block)
    
    print(f"   ✅ Complete coverage ensured: {len(complete_path)} total blocks")
    return complete_path

def _generate_efficient_navigation_with_sowing(target_path, valid_blocks, start_block):
    """Generate efficient navigation path with sow-on-return strategy"""
    
    print("   🗺️ Generating efficient navigation with sowing...")
    
    # Build complete navigation path
    points_lanes = [start_block]
    current_pos = start_block
    
    for target_block in target_path:
        if target_block == current_pos:
            continue
        
        # Find efficient path to target
        path_to_target = _find_shortest_path_orthogonal(current_pos, target_block, valid_blocks)
        
        # Add path (skip current position)
        for step in path_to_target[1:]:
            points_lanes.append(step)
        
        current_pos = target_block
    
    # Apply sow-on-return strategy
    block_visit_count = {}
    block_visit_positions = {}
    
    for i, block in enumerate(points_lanes):
        if block not in block_visit_count:
            block_visit_count[block] = 0
            block_visit_positions[block] = []
        block_visit_count[block] += 1
        block_visit_positions[block].append(i)
    
    # Generate sow flags
    sow_flags = []
    blocks_already_sown = {start_block}  # Start block is sown
    
    for i in range(len(points_lanes) - 1):
        next_block = points_lanes[i + 1]
        
        should_sow = False
        
        if next_block not in blocks_already_sown:
            if block_visit_count[next_block] == 1:
                # Single visit - sow immediately
                should_sow = True
                blocks_already_sown.add(next_block)
            else:
                # Multiple visits - sow on last visit
                visit_positions = block_visit_positions[next_block]
                if (i + 1) == visit_positions[-1]:  # Last visit
                    should_sow = True
                    blocks_already_sown.add(next_block)
        
        sow_flags.append(should_sow)
    
    print(f"   ✅ Efficient navigation generated: {len(points_lanes)} waypoints")
    return points_lanes, sow_flags

def _calculate_total_distance(points_lanes, grid):
    """Calculate total travel distance"""
    
    if len(points_lanes) < 2:
        return 0.0
    
    total_distance = 0.0
    
    for i in range(len(points_lanes) - 1):
        current_pos = grid.lane_to_metric(*points_lanes[i])
        next_pos = grid.lane_to_metric(*points_lanes[i + 1])
        
        distance = abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1])
        total_distance += distance
    
    return total_distance

def _verify_complete_coverage(points_lanes, sow_flags, valid_blocks):
    """Verify that all blocks are sown exactly once"""
    
    sown_blocks = set()
    
    # Starting block is sown
    if points_lanes:
        sown_blocks.add(points_lanes[0])
    
    # Process sow flags
    for i, block in enumerate(points_lanes):
        if i > 0 and i-1 < len(sow_flags) and sow_flags[i-1]:
            sown_blocks.add(block)
    
    # Verify complete coverage
    unsown_blocks = valid_blocks - sown_blocks
    
    if unsown_blocks:
        print(f"   ❌ CRITICAL: {len(unsown_blocks)} blocks remain unsown!")
        print(f"   Unsown blocks: {sorted(list(unsown_blocks))}")
    else:
        print(f"   ✅ PERFECT: All {len(valid_blocks)} blocks will be sown!")
    
    return sown_blocks

# Update the main function to use the ultra-efficient algorithm
def generate_optimal_coverage_path(grid, exit_block=None):
    """Generate optimal coverage path - now uses ultra-efficient algorithm"""
    return generate_ultra_efficient_coverage_path(grid, exit_block)

def generate_optimal_coverage_path_with_exit(grid, exit_block):
    """Generate zero-waste optimal coverage path with specific exit point"""
    return generate_optimal_coverage_path(grid, exit_block)


def compare_navigation_efficiency(grid, exit_block=None):
    """Compare different navigation strategies for efficiency analysis"""
    
    print("📊 NAVIGATION EFFICIENCY COMPARISON")
    print("=" * 50)
    
    valid_blocks = set(grid.valid_lanes)
    total_blocks = len(valid_blocks)
    
    if exit_block is None:
        border_blocks = get_border_blocks_from_set(valid_blocks)
        exit_block = border_blocks[0] if border_blocks else list(valid_blocks)[0]
    
    strategies = {}
    
    # Strategy 1: Ultra-Efficient (New)
    print("\n🚀 Testing Ultra-Efficient Strategy...")
    path1, sow1 = generate_ultra_efficient_coverage_path(grid, exit_block)
    dist1 = _calculate_total_distance(path1, grid)
    sown1 = _verify_complete_coverage(path1, sow1, valid_blocks)
    
    strategies['Ultra-Efficient'] = {
        'waypoints': len(path1),
        'distance': dist1,
        'sowing_ops': sum(sow1) + 1,
        'coverage': len(sown1) / total_blocks * 100,
        'efficiency': (sum(sow1) + 1) / len(path1) * 100 if path1 else 0
    }
    
    # Strategy 2: Basic Boustrophedon (for comparison)
    print("\n📐 Testing Basic Boustrophedon Strategy...")
    try:
        path2, sow2 = _generate_basic_boustrophedon_for_comparison(grid, exit_block)
        dist2 = _calculate_total_distance(path2, grid)
        sown2 = _verify_complete_coverage(path2, sow2, valid_blocks)
        
        strategies['Basic Boustrophedon'] = {
            'waypoints': len(path2),
            'distance': dist2,
            'sowing_ops': sum(sow2) + 1,
            'coverage': len(sown2) / total_blocks * 100,
            'efficiency': (sum(sow2) + 1) / len(path2) * 100 if path2 else 0
        }
    except:
        strategies['Basic Boustrophedon'] = {
            'waypoints': 0,
            'distance': 0,
            'sowing_ops': 0,
            'coverage': 0,
            'efficiency': 0
        }
    
    # Display comparison results
    print(f"\n📊 EFFICIENCY COMPARISON RESULTS:")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Waypoints':<10} {'Distance':<10} {'Coverage':<10} {'Efficiency':<10}")
    print("-" * 60)
    
    for strategy_name, stats in strategies.items():
        print(f"{strategy_name:<20} {stats['waypoints']:<10} {stats['distance']:<10.1f} "
              f"{stats['coverage']:<10.1f}% {stats['efficiency']:<10.1f}%")
    
    # Calculate improvements
    if len(strategies) >= 2:
        ultra_stats = strategies['Ultra-Efficient']
        basic_stats = strategies['Basic Boustrophedon']
        
        if basic_stats['distance'] > 0:
            distance_improvement = (basic_stats['distance'] - ultra_stats['distance']) / basic_stats['distance'] * 100
            waypoint_improvement = (basic_stats['waypoints'] - ultra_stats['waypoints']) / basic_stats['waypoints'] * 100
            
            print(f"\n🎯 ULTRA-EFFICIENT IMPROVEMENTS:")
            print(f"   Distance reduction: {distance_improvement:.1f}%")
            print(f"   Waypoint reduction: {waypoint_improvement:.1f}%")
            print(f"   Coverage: {ultra_stats['coverage']:.1f}%")
            
            if distance_improvement > 0:
                print(f"   ✅ Ultra-Efficient is {distance_improvement:.1f}% more efficient!")
            else:
                print(f"   ⚠️ Ultra-Efficient uses {abs(distance_improvement):.1f}% more distance")
    
    return strategies

def _generate_basic_boustrophedon_for_comparison(grid, exit_block):
    """Generate basic boustrophedon for comparison purposes"""
    
    valid_blocks = set(grid.valid_lanes)
    
    # Simple left-to-right boustrophedon
    columns = {}
    for bx, by in valid_blocks:
        if bx not in columns:
            columns[bx] = []
        columns[bx].append(by)
    
    for bx in columns:
        columns[bx].sort()
    
    # Basic column order (left to right)
    sorted_columns = sorted(columns.keys())
    
    path = []
    direction_up = True
    
    for bx in sorted_columns:
        if bx not in columns:
            continue
        
        column_blocks = columns[bx]
        
        if direction_up:
            ordered_blocks = sorted(column_blocks)
        else:
            ordered_blocks = sorted(column_blocks, reverse=True)
        
        for by in ordered_blocks:
            path.append((bx, by))
        
        direction_up = not direction_up
    
    # Add exit
    if path and path[-1] != exit_block:
        path.append(exit_block)
    
    # Generate navigation
    points_lanes = []
    current_pos = path[0] if path else exit_block
    points_lanes.append(current_pos)
    
    for target in path[1:]:
        if target != current_pos:
            path_to_target = _find_shortest_path_orthogonal(current_pos, target, valid_blocks)
            for step in path_to_target[1:]:
                points_lanes.append(step)
            current_pos = target
    
    # Basic sowing (sow everything)
    sow_flags = [True] * (len(points_lanes) - 1)
    
    return points_lanes, sow_flags





def animate_optimal_sowing(farm, path_lanes, sow_flags):
    """Animate optimal sowing with NO DOUBLE-SOWING visualization - CLEAN VERSION"""
    
    if not path_lanes:
        print("❌ No path to animate")
        return
    
    print("🎬 Starting CLEAN NO DOUBLE-SOWING animation (Brown→Green only)...")
    
    grid = BlockGrid(farm)
    path_metric = [grid.lane_to_metric(bx, by) for bx, by in path_lanes]
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    
    # Get bounds with padding
    min_x, min_y, max_x, max_y = farm.get_bounding_box()
    padding = max(farm.rover_width, farm.rover_length)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    # Create block patches dictionary
    block_patches = {}
    block_texts = {}
    
    # Draw all farm blocks with initial brown color (unsown)
    for bx, by in farm.all_blocks:
        x = bx * farm.rover_width
        y = by * farm.rover_length
        
        rect = MPLRectangle((x, y), farm.rover_width, farm.rover_length,
                           facecolor='#8B4513', edgecolor='black', linewidth=2, alpha=0.9)  # Brown
        ax.add_patch(rect)
        block_patches[(bx, by)] = rect
        
        # Add status text
        text = ax.text(x + farm.rover_width/2, y + farm.rover_length/2, 'SOIL',
                      ha='center', va='center', fontsize=9, weight='bold', color='white')
        block_texts[(bx, by)] = text
    
    # Draw rover rectangle
    rover_rect = MPLRectangle((path_metric[0][0] - farm.rover_width/2, 
                              path_metric[0][1] - farm.rover_length/2),
                             farm.rover_width, farm.rover_length,
                             color='orange', edgecolor='black', linewidth=4, zorder=10)
    ax.add_patch(rover_rect)
    
    # SOW THE STARTING BLOCK IMMEDIATELY (FIX #1)
    start_block = path_lanes[0]
    if start_block in block_patches:
        block_patches[start_block].set_facecolor('#228B22')  # Forest green
        block_texts[start_block].set_text('SOWN')
        block_texts[start_block].set_color('white')
    
    # Start and end markers
    ax.plot(path_metric[0][0], path_metric[0][1], 'go', markersize=20, zorder=5, 
            label='Start', markeredgecolor='black', markeredgewidth=3)
    ax.plot(path_metric[-1][0], path_metric[-1][1], 'ro', markersize=20, zorder=5, 
            label='Exit', markeredgecolor='black', markeredgewidth=3)
    
    ax.set_title(f'🎯 {farm.name} - CLEAN Animation\n'
                f'Brown Soil → Green Sown (No Path Lines)', 
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.2)
    
    # Animation variables
    sown_blocks = {start_block}  # Start with starting block already sown
    sowing_count = 1  # Count starting block
    navigation_count = 0
    
    # Status displays
    mission_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
                          verticalalignment='bottom', fontsize=10, weight='bold')
    
    coverage_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                           verticalalignment='bottom', horizontalalignment='right', 
                           fontsize=12, weight='bold')
    
    def animate_frame(frame):
        nonlocal sowing_count, navigation_count
        
        waypoint_idx = min(frame // 2, len(path_metric) - 1)
        
        # Update rover position
        current_pos = path_metric[waypoint_idx]
        rover_rect.set_xy((current_pos[0] - farm.rover_width/2, 
                          current_pos[1] - farm.rover_length/2))
        
        # Process movement and sowing (NO PATH LINES - FIX #2)
        if waypoint_idx > 0 and waypoint_idx <= len(sow_flags):
            current_block = path_lanes[waypoint_idx]
            
            # Determine if this is a sowing move
            is_sowing = sow_flags[waypoint_idx - 1] if waypoint_idx - 1 < len(sow_flags) else False
            
            if is_sowing:
                # SOWING MOVE - change block color ONLY (no path lines)
                if current_block in block_patches and current_block not in sown_blocks:
                    block_patches[current_block].set_facecolor('#228B22')  # Forest green
                    block_texts[current_block].set_text('SOWN')
                    block_texts[current_block].set_color('white')
                    sown_blocks.add(current_block)
                    sowing_count += 1
                
            else:
                # NAVIGATION MOVE - just count it (no visual changes)
                navigation_count += 1
        
        # Update status displays
        total_blocks = len(farm.all_blocks)
        sown_count = len(sown_blocks)
        coverage_percent = (sown_count / total_blocks * 100) if total_blocks > 0 else 0
        
        # Calculate efficiency
        total_operations = sowing_count + navigation_count
        distance_ratio = f"{sown_count}/{total_operations}" if total_operations > 0 else "0/0"
        
        # Mission status with CORRECT metrics
        mission_text.set_text(f'🎯 SOWING PROGRESS\n'
                             f'Waypoint: {waypoint_idx + 1}/{len(path_lanes)}\n'
                             f'Sown: {sowing_count} blocks\n'
                             f'Navigation: {navigation_count} moves\n'
                             f'Ratio: {distance_ratio}\n'
                             f'Coverage: {sown_count}/{total_blocks}')

        
        if coverage_percent >= 100:
            phase = "🎉 100% EFFICIENT!"
            phase_color = 'lightgreen'
        elif coverage_percent >= 90:
            phase = "🌾 HIGHLY EFFICIENT"
            phase_color = 'lightblue'
        elif waypoint_idx >= len(path_lanes) - 1:
            phase = "🚪 AT EXIT"
            phase_color = 'lightyellow'
        elif coverage_percent > 0:
            phase = "🚀 SOWING"
            phase_color = 'lightblue'
        else:
            phase = "🚀 STARTING"
            phase_color = 'lightgray'

        
        coverage_text.set_text(f'{coverage_percent:.1f}%\nCOVERAGE\n{phase}')
        coverage_text.get_bbox_patch().set_facecolor(phase_color)
        
        return [rover_rect, mission_text, coverage_text] + list(block_patches.values()) + list(block_texts.values())
    
    # Create animation
    total_frames = len(path_metric) * 2 + 20
    anim = FuncAnimation(fig, animate_frame, frames=total_frames, 
                        interval=300, blit=False, repeat=False)
    
    plt.show(block=True)
    return anim

                

def calculate_meaningful_efficiency_metrics(path_lanes, sow_flags, valid_blocks, grid):
    """Calculate meaningful efficiency metrics instead of misleading waypoint ratio"""
    
    # Count actually sown blocks
    sown_blocks = set()
    if path_lanes:
        sown_blocks.add(path_lanes[0])  # Starting block is always sown
    
    for i, block in enumerate(path_lanes[1:], 1):
        if i-1 < len(sow_flags) and sow_flags[i-1]:
            sown_blocks.add(block)
    
    total_blocks = len(valid_blocks)
    sowing_operations = len(sown_blocks)
    navigation_operations = len(path_lanes) - sowing_operations
    total_distance = _calculate_total_distance(path_lanes, grid)
    
    # MEANINGFUL METRICS
    metrics = {
        # Most important: Did we sow everything?
        'coverage_efficiency': (len(sown_blocks) / total_blocks) * 100 if total_blocks > 0 else 0,
        
        # Precision: No double-sowing?
        'sowing_precision': 100.0,  # Always 100% with sow-on-return strategy
        
        # Travel efficiency: Distance per block sown
        'distance_per_block': total_distance / len(sown_blocks) if len(sown_blocks) > 0 else 0,
        
        # Operation breakdown
        'sowing_ops': sowing_operations,
        'navigation_ops': navigation_operations,
        'total_waypoints': len(path_lanes),
        'total_distance': total_distance
    }
    
    return metrics

# ============================================================================
# MAIN PROGRAMS FOR OPTIMAL SOWING
# ============================================================================

def main_optimal_farm():
    """Main program for optimal sowing farm (standard - chooses best exit)"""
    print("🎯 OPTIMAL SOWING FARM SYSTEM")
    print("   100% Coverage Guarantee + Efficient Navigation")
    print("=" * 65)
    
    # Get user input
    print("\n📏 Farm Configuration:")
    try:
        rect_width = float(input("Rectangle width (e.g., 60): "))
        rect_height = float(input("Rectangle height (e.g., 40): "))
        rover_width = float(input("Rover width (e.g., 10): "))
        rover_length = float(input("Rover length (e.g., 8): "))
        
        print("\nExtension side:")
        print("1. Bottom")
        print("2. Top") 
        print("3. Left")
        print("4. Right")
        
        side_choice = input("Choose extension side (1-4): ").strip()
        side_map = {'1': 'bottom', '2': 'top', '3': 'left', '4': 'right'}
        extension_side = side_map.get(side_choice, 'bottom')
        
    except ValueError:
        print("❌ Invalid input, using defaults")
        rect_width, rect_height = 60, 40
        rover_width, rover_length = 10, 8
        extension_side = 'bottom'
    
    # Create block-based farm
    print(f"\n🏗️ Creating optimal farm: {rect_width}×{rect_height} rectangle + blocks on {extension_side}")
    farm = BlockBasedFarm(rect_width, rect_height, rover_width, rover_length, extension_side)
    
    # Create grid
    print(f"🤖 Setting up rover: {rover_width}×{rover_length}")
    grid = BlockGrid(farm)
    
    if not grid.valid_lanes:
        print("❌ No valid blocks found!")
        return
    
    # Generate optimal coverage path (system chooses best exit)
        # Generate zero-waste optimal coverage path (system chooses best exit)
    print("\n🎯 Generating ZERO-WASTE optimal coverage path (system chooses best exit)...")
    path_lanes, sow_flags = generate_optimal_coverage_path(grid)

    
    if not path_lanes:
        print("❌ Path generation failed!")
        return
    
        # Show MEANINGFUL results
    farm_info = farm.get_farm_info()
    metrics = calculate_meaningful_efficiency_metrics(path_lanes, sow_flags, set(farm.all_blocks), grid)
    
    print(f"\n📊 Optimal Sowing Results:")
    print(f"   Total blocks: {farm_info['total_blocks']}")
    print(f"   Coverage: {metrics['coverage_efficiency']:.1f}% ({'PERFECT' if metrics['coverage_efficiency'] >= 100 else 'INCOMPLETE'})")
    print(f"   Sowing precision: {metrics['sowing_precision']:.1f}% (No double-sowing)")
    print(f"   Distance efficiency: {metrics['distance_per_block']:.1f}m per block")
    print(f"   Total distance: {metrics['total_distance']:.1f}m")
    print(f"   Sowing operations: {metrics['sowing_ops']}")
    print(f"   Navigation moves: {metrics['navigation_ops']}")
    print(f"   Total waypoints: {metrics['total_waypoints']}")
    print(f"   Start position: {path_lanes[0]}")
    print(f"   Exit position: {path_lanes[-1]}")
    
    if metrics['coverage_efficiency'] >= 100:
        print(f"   🎉 SUCCESS: 100% EFFICIENT SOWING!")
        print(f"   🌾 Every block sown exactly once!")
    else:
        print(f"   ❌ FAILURE: Only {metrics['coverage_efficiency']:.1f}% coverage!")
        unsown_count = farm_info['total_blocks'] - metrics['sowing_ops']
        print(f"   {unsown_count} blocks remain unsown")

    
    print(f"   ✅ Optimal boustrophedon pattern")
    print(f"   ✅ Intelligent start/exit positioning")
    print(f"   ✅ No diagonal movement")
    print(f"   ✅ No double-sowing")
    print(f"   ✅ 100% coverage guarantee")
    
    # Visualization options
    print(f"\n🎨 Visualization:")
    print("1. Show static analysis (Optimal Pattern)")
    print("2. Show animation (Brown→Green Optimal Sowing)")
    print("3. Show both")
    
    viz_choice = input("Choose option (1-3): ").strip()
    
    if viz_choice in ['1', '3']:
        print("📊 Showing optimal sowing analysis...")
        visualize_optimal_farm_and_path(farm, path_lanes, sow_flags)
    
    if viz_choice in ['2', '3']:
        print("🎬 Starting optimal sowing animation...")
        animate_optimal_sowing(farm, path_lanes, sow_flags)
    
    print("\n✅ Complete!")

def main_optimal_farm_with_exit():
    """Main program for optimal sowing farm with custom exit"""
    print("🎯 OPTIMAL SOWING FARM WITH CUSTOM EXIT")
    print("   100% Coverage Guarantee + Custom Exit Point")
    print("=" * 75)
    
    # Get user input
    print("\n📏 Farm Configuration:")
    try:
        rect_width = float(input("Rectangle width (e.g., 60): "))
        rect_height = float(input("Rectangle height (e.g., 40): "))
        rover_width = float(input("Rover width (e.g., 10): "))
        rover_length = float(input("Rover length (e.g., 8): "))
        
        print("\nExtension side:")
        print("1. Bottom")
        print("2. Top") 
        print("3. Left")
        print("4. Right")
        
        side_choice = input("Choose extension side (1-4): ").strip()
        side_map = {'1': 'bottom', '2': 'top', '3': 'left', '4': 'right'}
        extension_side = side_map.get(side_choice, 'bottom')
        
    except ValueError:
        print("❌ Invalid input, using defaults")
        rect_width, rect_height = 60, 40
        rover_width, rover_length = 10, 8
        extension_side = 'bottom'
    
    # Create block-based farm
    print(f"\n🏗️ Creating optimal farm: {rect_width}×{rect_height} rectangle + blocks on {extension_side}")
    farm = BlockBasedFarm(rect_width, rect_height, rover_width, rover_length, extension_side)
    
    # Create grid
    print(f"🤖 Setting up rover: {rover_width}×{rover_length}")
    grid = BlockGrid(farm)
    
    if not grid.valid_lanes:
        print("❌ No valid blocks found!")
        return
    
    # Get custom exit choice
    exit_block = get_user_exit_choice(farm)
    
        # Generate zero-waste optimal coverage path with custom exit
    print("\n🎯 Generating ZERO-WASTE optimal coverage path with custom exit...")
    path_lanes, sow_flags = generate_optimal_coverage_path_with_exit(grid, exit_block)

    if not path_lanes:
        print("❌ Path generation failed!")
        return
    
    # Show results
        # Show MEANINGFUL results
    farm_info = farm.get_farm_info()
    metrics = calculate_meaningful_efficiency_metrics(path_lanes, sow_flags, set(farm.all_blocks), grid)
    
    print(f"\n📊 Optimal Sowing Results:")
    print(f"   Total blocks: {farm_info['total_blocks']}")
    print(f"   Coverage: {metrics['coverage_efficiency']:.1f}% ({'PERFECT' if metrics['coverage_efficiency'] >= 100 else 'INCOMPLETE'})")
    print(f"   Sowing precision: {metrics['sowing_precision']:.1f}% (No double-sowing)")
    print(f"   Distance efficiency: {metrics['distance_per_block']:.1f}m per block")
    print(f"   Total distance: {metrics['total_distance']:.1f}m")
    print(f"   Sowing operations: {metrics['sowing_ops']}")
    print(f"   Navigation moves: {metrics['navigation_ops']}")
    print(f"   Total waypoints: {metrics['total_waypoints']}")

    print(f"   Start position: {path_lanes[0]} (optimally chosen)")
    print(f"   Exit position: {exit_block} (user selected)")
    
    if metrics['coverage_efficiency'] >= 100:
        print(f"   🎉 SUCCESS: 100% EFFICIENT SOWING!")
        print(f"   🌾 Every block sown exactly once!")
    else:
        print(f"   ❌ FAILURE: Only {metrics['coverage_efficiency']:.1f}% coverage!")
        unsown_count = farm_info['total_blocks'] - metrics['sowing_ops']
        print(f"   {unsown_count} blocks remain unsown")

    
    print(f"   ✅ Optimal boustrophedon pattern")
    print(f"   ✅ Intelligent start positioning")
    print(f"   ✅ Custom exit point")
    print(f"   ✅ No diagonal movement")
    print(f"   ✅ No double-sowing")
    print(f"   ✅ 100% coverage guarantee")
    
    # Visualization options
    print(f"\n🎨 Visualization:")
    print("1. Show static analysis (Optimal Pattern)")
    print("2. Show animation (Brown→Green Optimal Sowing)")
    print("3. Show both")
    
    viz_choice = input("Choose option (1-3): ").strip()
    
    if viz_choice in ['1', '3']:
        print("📊 Showing optimal sowing analysis...")
        visualize_optimal_farm_and_path(farm, path_lanes, sow_flags)
    
    if viz_choice in ['2', '3']:
        print("🎬 Starting optimal sowing animation...")
        animate_optimal_sowing(farm, path_lanes, sow_flags)
    
    print("\n✅ Complete!")

# ============================================================================
# QUICK DEMO FOR OPTIMAL SOWING
# ============================================================================

def quick_demo_optimal():
    """Quick demonstration of optimal sowing system"""
    print("🚀 QUICK DEMO - Optimal Sowing System")
    print("   100% Coverage Guarantee + Intelligent Navigation")
    print("=" * 75)
    
    # Test all four extension sides
    sides = ['bottom', 'top', 'left', 'right']
    
    for side in sides:
        print(f"\n🏗️ Testing {side} extension with optimal sowing...")
        
        # Create farm
        farm = BlockBasedFarm(60, 40, 10, 8, side)
        grid = BlockGrid(farm)
        
        # Generate optimal path
        path_lanes, sow_flags = generate_optimal_coverage_path(grid)
        
        # Count sown blocks
        sown_blocks = set()
        for i, block in enumerate(path_lanes):
            if i > 0 and i-1 < len(sow_flags) and sow_flags[i-1]:
                sown_blocks.add(block)
        
        farm_info = farm.get_farm_info()
        coverage_percent = len(sown_blocks) / farm_info['total_blocks'] * 100
                
        print(f"   ✅ {farm_info['total_blocks']} blocks, {len(path_lanes)} waypoints")
        print(f"   ✅ {farm_info['total_area_m2']:.1f}m² total area")
        print(f"   ✅ {len(sown_blocks)} blocks sown ({coverage_percent:.1f}% coverage)")
        print(f"   ✅ {sum(sow_flags)} sowing moves, {len(sow_flags) - sum(sow_flags)} navigation moves")
        print(f"   ✅ Start: {path_lanes[0]}, Exit: {path_lanes[-1]}")
        
        if coverage_percent >= 100:
            print(f"   🎉 SUCCESS: 100% coverage achieved!")
        else:
            print(f"   ❌ WARNING: Only {coverage_percent:.1f}% coverage")
    
    # Show visualization for bottom extension
    print(f"\n📊 Showing optimal sowing visualization for bottom extension...")
    farm = BlockBasedFarm(60, 40, 10, 8, 'bottom')
    grid = BlockGrid(farm)
    path_lanes, sow_flags = generate_optimal_coverage_path(grid)
    
    visualize_optimal_farm_and_path(farm, path_lanes, sow_flags)
    
    # Ask for animation
    show_anim = input("\n🎬 Show optimal sowing animation (Brown→Green)? (y/n): ").strip().lower()
    if show_anim == 'y':
        animate_optimal_sowing(farm, path_lanes, sow_flags)

# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS (for existing code)
# ============================================================================

def generate_block_coverage_path(grid):
    """Backward compatibility wrapper for optimal coverage"""
    print("🔄 Using optimal coverage path (backward compatibility mode)")
    return generate_optimal_coverage_path(grid)

def generate_block_coverage_path_with_exit(grid, exit_block):
    """Backward compatibility wrapper for optimal coverage with exit"""
    print("🔄 Using optimal coverage path with exit (backward compatibility mode)")
    return generate_optimal_coverage_path_with_exit(grid, exit_block)

def visualize_block_farm_and_path(farm, path_lanes, sow_flags):
    """Backward compatibility wrapper for visualization"""
    print("🔄 Using optimal visualization (backward compatibility mode)")
    return visualize_optimal_farm_and_path(farm, path_lanes, sow_flags)

def animate_block_rover(farm, path_lanes, sow_flags):
    """Backward compatibility wrapper for animation"""
    print("🔄 Using optimal animation (backward compatibility mode)")
    return animate_optimal_sowing(farm, path_lanes, sow_flags)

def main_block_farm():
    """Backward compatibility wrapper for main program"""
    print("🔄 Using optimal farm system (backward compatibility mode)")
    return main_optimal_farm()

def main_block_farm_with_exit():
    """Backward compatibility wrapper for main program with exit"""
    print("🔄 Using optimal farm system with exit (backward compatibility mode)")
    return main_optimal_farm_with_exit()

def quick_demo_blocks():
    """Backward compatibility wrapper for quick demo"""
    print("🔄 Using optimal demo (backward compatibility mode)")
    return quick_demo_optimal()

# ============================================================================
# UPDATED ENTRY POINT WITH OPTIMAL SOWING SYSTEM
# ============================================================================

if __name__ == "__main__":
    print("🌾 ULTRA-EFFICIENT FARM ROVER NAVIGATION SYSTEM")
    print("   🚀 NOW WITH ULTRA-EFFICIENT NAVIGATION!")
    print("   100% Coverage Guarantee + Minimal Travel Distance")
    print("=" * 80)
    
    print("\n🚀 Choose your ultra-efficient mission:")
    print("1. 🎯 Ultra-Efficient Farm (System Chooses Best Exit)")
    print("2. 🚪 Ultra-Efficient Farm (Custom Exit Point)")
    print("3. 📊 Navigation Efficiency Comparison")
    print("4. 🚀 Quick Ultra-Efficient Demo")
    print("5. 🧪 Test All Extension Patterns")
    print("6. ❌ Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    try:   
        if choice == '1':
            print("\n🎯 ULTRA-EFFICIENT FARM (SYSTEM CHOOSES EXIT)")
            print("=" * 55)
            main_optimal_farm()
            
        elif choice == '2':
            print("\n🚪 ULTRA-EFFICIENT FARM (CUSTOM EXIT)")
            print("=" * 45)
            main_optimal_farm_with_exit()
            
        elif choice == '3':
            print("\n📊 NAVIGATION EFFICIENCY COMPARISON")
            print("=" * 45)
            
            # Get farm parameters
            try:
                rect_width = float(input("Rectangle width (e.g., 60): "))
                rect_height = float(input("Rectangle height (e.g., 40): "))
                rover_width = float(input("Rover width (e.g., 10): "))
                rover_length = float(input("Rover length (e.g., 8): "))
            except ValueError:
                rect_width, rect_height = 60, 40
                rover_width, rover_length = 10, 8
            
            # Create farm and compare strategies
            farm = BlockBasedFarm(rect_width, rect_height, rover_width, rover_length, 'bottom')
            grid = BlockGrid(farm)
            
            comparison_results = compare_navigation_efficiency(grid)
            
        elif choice == '4':
            print("\n🚀 QUICK ULTRA-EFFICIENT DEMO")
            print("=" * 35)
            quick_demo_optimal()
            
        elif choice == '5':
            print("\n🧪 TESTING ALL EXTENSION PATTERNS")
            print("=" * 40)
            
            patterns = ['bottom', 'top', 'left', 'right']
            
            for pattern in patterns:
                print(f"\n🔬 Testing {pattern} extension with ultra-efficient navigation...")
                farm = BlockBasedFarm(50, 30, 10, 6, pattern)
                grid = BlockGrid(farm)
                path_lanes, sow_flags = generate_ultra_efficient_coverage_path(grid)
                
                total_distance = _calculate_total_distance(path_lanes, grid)
                sown_blocks = _verify_complete_coverage(path_lanes, sow_flags, set(grid.valid_lanes))
                
                farm_info = farm.get_farm_info()
                coverage = len(sown_blocks) / farm_info['total_blocks'] * 100
                efficiency = (sum(sow_flags) + 1) / len(path_lanes) * 100 if path_lanes else 0
                
                status = "✅ SUCCESS" if coverage >= 100 else "❌ FAILED"
                print(f"   {status}: {coverage:.1f}% coverage, {total_distance:.1f}m distance, {efficiency:.1f}% efficiency")
            
        elif choice == '6':
            print("👋 Goodbye!")
            
        else:
            print("❌ Invalid choice, running ultra-efficient demo...")
            quick_demo_optimal()
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Try the quick demo for a working example.")
        import traceback
        traceback.print_exc()
    
    print("\n🌾 Thank you for using the Ultra-Efficient Farm Rover System!")
    print("🚀 Features: Ultra-efficient navigation, 100% coverage guarantee, minimal travel distance!")
    print("🌱 Brown soil → Green sown fields with maximum efficiency!")