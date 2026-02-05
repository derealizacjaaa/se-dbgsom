
from pathlib import Path

try:
    print("Importing juliacall...")
    from juliacall import Main as jl
    print("juliacall imported successfully.")
    
    project_root = Path(__file__).parent
    basic_path = project_root / "src" / "DBGSOM.jl"
    basic_path_str = str(basic_path).replace('\\', '/')
    
    print(f"Loading DBGSOM from {basic_path_str}...")
    
    jl.seval('using Random')
    jl.seval('using Statistics')
    
    # Try to include the file
    jl.seval(f'include("{basic_path_str}")')
    print("Included DBGSOM.jl")
    
    jl.seval('using .BasicDBGSOM')
    print("Used .BasicDBGSOM")
    
except Exception as e:
    print("Error:", e)
