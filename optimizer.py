from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class RakeOptimizer:
    """
    Enhanced rake formation optimizer with multiple objectives and constraints.
    Optimizes for wagon utilization, demand priority, and operational efficiency.
    """
    
    def __init__(self, 
                 target_utilization: float = 0.85,
                 min_utilization: float = 0.50,
                 max_utilization: float = 0.95,
                 time_limit: int = 300):
        """
        Initialize optimizer with configurable parameters.
        
        Args:
            target_utilization: Ideal wagon capacity usage (0-1)
            min_utilization: Minimum acceptable utilization
            max_utilization: Maximum safe utilization (avoid overloading)
            time_limit: Maximum solving time in seconds
        """
        self.target_utilization = target_utilization
        self.min_utilization = min_utilization
        self.max_utilization = max_utilization
        self.time_limit = time_limit
        self.solve_status = None
        self.solve_time = 0
        self.objective_value = None
    
    def optimize_rake_plan(self, 
                          orders: Dict[str, float], 
                          wagons: Dict[str, float],
                          demand_priorities: Optional[Dict[str, int]] = None,
                          material_compatibility: Optional[Dict[str, List[str]]] = None,
                          optimize_for: str = 'balanced') -> List[Tuple[str, str]]:
        """
        Optimize rake formation with advanced constraints.
        
        Args:
            orders: Dict of {order_id: weight}
            wagons: Dict of {wagon_id: capacity}
            demand_priorities: Dict of {order_id: priority_score} (1-5, 5=highest)
            material_compatibility: Dict of {wagon_id: [allowed_materials]}
            optimize_for: 'utilization', 'priority', or 'balanced'
        
        Returns:
            List of (order_id, wagon_id) assignments
        """
        start_time = time.time()
        
        # Validate inputs
        if not orders or not wagons:
            print("⚠️ Error: Empty orders or wagons dictionary")
            return []
        
        # Check if problem is feasible
        total_order_weight = sum(orders.values())
        total_wagon_capacity = sum(wagons.values())
        
        if total_order_weight > total_wagon_capacity:
            print(f"⚠️ Warning: Total order weight ({total_order_weight:.2f}) exceeds total capacity ({total_wagon_capacity:.2f})")
            print("   Some orders may not be assigned.")
        
        # Create problem
        if optimize_for == 'utilization':
            prob = LpProblem("RakeFormation_Utilization", LpMaximize)
        else:
            prob = LpProblem("RakeFormation_Balanced", LpMinimize)
        
        # Decision variables
        assign = LpVariable.dicts("Assign", 
                                 [(o, w) for o in orders for w in wagons], 
                                 cat='Binary')
        
        # Auxiliary variables for utilization tracking
        wagon_used = LpVariable.dicts("WagonUsed", wagons, cat='Binary')
        wagon_load = LpVariable.dicts("WagonLoad", wagons, lowBound=0)
        utilization_deviation = LpVariable.dicts("UtilDev", wagons, lowBound=0)
        
        # Set default priorities if not provided
        if demand_priorities is None:
            demand_priorities = {o: 3 for o in orders}  # Default medium priority
        
        # Normalize priorities to 0-1 scale
        max_priority = max(demand_priorities.values()) if demand_priorities else 5
        normalized_priorities = {o: demand_priorities.get(o, 3) / max_priority 
                                for o in orders}
        
        # === OBJECTIVE FUNCTION ===
        if optimize_for == 'utilization':
            # Maximize overall utilization
            prob += lpSum([wagon_load[w] / wagons[w] * wagon_used[w] for w in wagons])
        
        elif optimize_for == 'priority':
            # Minimize weighted deviation, prioritizing high-demand orders
            prob += lpSum([
                assign[o, w] * orders[o] * (1 - normalized_priorities[o]) 
                for o in orders for w in wagons
            ])
        
        else:  # balanced
            # Multi-objective: minimize wagons used + utilization deviation + priority penalty
            wagon_count_weight = 100.0
            utilization_weight = 10.0
            priority_weight = 1.0
            
            prob += (
                wagon_count_weight * lpSum([wagon_used[w] for w in wagons]) +
                utilization_weight * lpSum([utilization_deviation[w] for w in wagons]) +
                priority_weight * lpSum([
                    assign[o, w] * (1 - normalized_priorities[o]) 
                    for o in orders for w in wagons
                ])
            )
        
        # === CONSTRAINTS ===
        
        # 1. Each order must be assigned to exactly one wagon (or not assigned if infeasible)
        for o in orders:
            prob += lpSum([assign[o, w] for w in wagons]) <= 1, f"Order_{o}_assignment"
        
        # 2. Wagon capacity constraint (with safety margin)
        for w in wagons:
            prob += (
                lpSum([assign[o, w] * orders[o] for o in orders]) <= 
                wagons[w] * self.max_utilization,
                f"Wagon_{w}_capacity"
            )
            
            # Link wagon_load to actual load
            prob += wagon_load[w] == lpSum([assign[o, w] * orders[o] for o in orders])
        
        # 3. Minimum utilization constraint (avoid nearly empty wagons)
        for w in wagons:
            prob += (
                wagon_load[w] >= wagons[w] * self.min_utilization * wagon_used[w],
                f"Wagon_{w}_min_utilization"
            )
        
        # 4. Link wagon_used to assignments
        for w in wagons:
            prob += (
                wagon_used[w] * len(orders) >= lpSum([assign[o, w] for o in orders]),
                f"Wagon_{w}_usage_link"
            )
        
        # 5. Track utilization deviation from target
        for w in wagons:
            target_load = wagons[w] * self.target_utilization
            prob += (
                utilization_deviation[w] >= wagon_load[w] - target_load,
                f"Wagon_{w}_deviation_pos"
            )
            prob += (
                utilization_deviation[w] >= target_load - wagon_load[w],
                f"Wagon_{w}_deviation_neg"
            )
        
        # 6. Material compatibility constraints (if provided)
        if material_compatibility:
            for o in orders:
                for w in wagons:
                    # If wagon has material restrictions and order material is not compatible
                    if w in material_compatibility and len(material_compatibility[w]) > 0:
                        # This would require material info in orders - skip for now
                        # Could be implemented with additional order metadata
                        pass
        
        # === SOLVE ===
        solver = PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)
        prob.solve(solver)
        
        self.solve_time = time.time() - start_time
        self.solve_status = LpStatus[prob.status]
        self.objective_value = prob.objective.value()
        
        # === EXTRACT RESULTS ===
        result = []
        unassigned_orders = []
        
        for o in orders:
            assigned = False
            for w in wagons:
                if assign[o, w].varValue and assign[o, w].varValue > 0.5:
                    result.append((o, w))
                    assigned = True
                    break
            if not assigned:
                unassigned_orders.append(o)
        
        # Report unassigned orders
        if unassigned_orders:
            print(f"⚠️ Warning: {len(unassigned_orders)} orders could not be assigned:")
            print(f"   {', '.join(unassigned_orders[:5])}" + 
                  (f" and {len(unassigned_orders)-5} more..." if len(unassigned_orders) > 5 else ""))
        
        return result
    
    def get_optimization_stats(self) -> Dict:
        """Return statistics about the last optimization run."""
        return {
            'status': self.solve_status,
            'solve_time': f"{self.solve_time:.2f}s",
            'objective_value': self.objective_value
        }
    
    def analyze_solution(self, result: List[Tuple[str, str]], 
                        orders: Dict[str, float], 
                        wagons: Dict[str, float]) -> Dict:
        """
        Analyze the quality of the solution.
        
        Returns comprehensive metrics about the assignment.
        """
        if not result:
            return {
                'error': 'No solution to analyze',
                'wagons_used': 0,
                'orders_assigned': 0
            }
        
        # Group by wagon
        wagon_assignments = {}
        for order_id, wagon_id in result:
            wagon_assignments.setdefault(wagon_id, []).append(order_id)
        
        # Calculate metrics
        utilizations = []
        total_weight_used = 0
        total_capacity_available = 0
        overloaded_wagons = 0
        underutilized_wagons = 0
        
        for wagon_id, assigned_orders in wagon_assignments.items():
            wagon_capacity = wagons[wagon_id]
            wagon_load = sum(orders[o] for o in assigned_orders)
            utilization = wagon_load / wagon_capacity
            
            utilizations.append(utilization)
            total_weight_used += wagon_load
            total_capacity_available += wagon_capacity
            
            if utilization > self.max_utilization:
                overloaded_wagons += 1
            elif utilization < self.min_utilization:
                underutilized_wagons += 1
        
        return {
            'wagons_used': len(wagon_assignments),
            'total_wagons_available': len(wagons),
            'orders_assigned': len(result),
            'total_orders': len(orders),
            'avg_utilization': f"{np.mean(utilizations)*100:.2f}%" if utilizations else "0%",
            'min_utilization': f"{np.min(utilizations)*100:.2f}%" if utilizations else "0%",
            'max_utilization': f"{np.max(utilizations)*100:.2f}%" if utilizations else "0%",
            'std_utilization': f"{np.std(utilizations)*100:.2f}%" if utilizations else "0%",
            'overloaded_wagons': overloaded_wagons,
            'underutilized_wagons': underutilized_wagons,
            'total_weight_assigned': f"{total_weight_used:.2f} tons",
            'total_capacity_used': f"{total_capacity_available:.2f} tons",
            'overall_efficiency': f"{(total_weight_used/total_capacity_available)*100:.2f}%" if total_capacity_available > 0 else "0%"
        }


# Backward compatible function
def optimize_rake_plan(orders: Dict[str, float], 
                       wagons: Dict[str, float],
                       **kwargs) -> List[Tuple[str, str]]:
    """
    Backward compatible optimization function.
    
    Additional kwargs:
        - demand_priorities: Dict of order priorities
        - optimize_for: 'utilization', 'priority', or 'balanced'
        - target_utilization: Target wagon fill rate (0-1)
    """
    optimizer = RakeOptimizer(
        target_utilization=kwargs.get('target_utilization', 0.85),
        min_utilization=kwargs.get('min_utilization', 0.50),
        max_utilization=kwargs.get('max_utilization', 0.95)
    )
    
    result = optimizer.optimize_rake_plan(
        orders=orders,
        wagons=wagons,
        demand_priorities=kwargs.get('demand_priorities'),
        optimize_for=kwargs.get('optimize_for', 'balanced')
    )
    
    return result


def optimize_with_analysis(orders: Dict[str, float], 
                           wagons: Dict[str, float],
                           **kwargs) -> Tuple[List[Tuple[str, str]], Dict, Dict]:
    """
    Optimize and return results with comprehensive analysis.
    
    Returns:
        - result: List of (order_id, wagon_id) assignments
        - stats: Optimization statistics
        - analysis: Solution quality metrics
    """
    optimizer = RakeOptimizer(
        target_utilization=kwargs.get('target_utilization', 0.85),
        min_utilization=kwargs.get('min_utilization', 0.50),
        max_utilization=kwargs.get('max_utilization', 0.95)
    )
    
    result = optimizer.optimize_rake_plan(
        orders=orders,
        wagons=wagons,
        demand_priorities=kwargs.get('demand_priorities'),
        optimize_for=kwargs.get('optimize_for', 'balanced')
    )
    
    stats = optimizer.get_optimization_stats()
    analysis = optimizer.analyze_solution(result, orders, wagons)
    
    return result, stats, analysis


if __name__ == "__main__":
    # Example usage and testing
    sample_orders = {
        'O1': 25.0,
        'O2': 18.0,
        'O3': 30.0,
        'O4': 12.0,
        'O5': 22.0
    }
    
    sample_wagons = {
        'W1': 50.0,
        'W2': 45.0,
        'W3': 40.0
    }
    
    sample_priorities = {
        'O1': 5,  # High priority
        'O2': 3,  # Medium
        'O3': 5,  # High
        'O4': 2,  # Low
        'O5': 4   # Medium-High
    }
    
    print("=== Testing Enhanced Rake Optimizer ===\n")
    
    # Test balanced optimization
    result, stats, analysis = optimize_with_analysis(
        orders=sample_orders,
        wagons=sample_wagons,
        demand_priorities=sample_priorities,
        optimize_for='balanced'
    )
    
    print(f"Optimization Status: {stats['status']}")
    print(f"Solve Time: {stats['solve_time']}")
    print(f"\nAssignments: {result}")
    print(f"\nAnalysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")