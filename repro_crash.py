
try:
    print("Importing juliacall...")
    from juliacall import Main as jl
    print("juliacall imported successfully.")
    print("Julia version:", jl.seval("VERSION"))
except Exception as e:
    print("Error importing juliacall:", e)
