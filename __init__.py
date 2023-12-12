# Imports
from main import main
from isaacgym import gymutil


if __name__ == "__main__":
    # Getting the arguments
    args = gymutil.parse_arguments(
    description="RMP based Constrained Motion Planner",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])
    
    # Calling main function
    main(args)