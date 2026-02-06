X_new = transport.step(
    X_old,          # shape: (n_shells, n_species)
    structure,      # dict-like container of profiles
    dt              # timestep
)

