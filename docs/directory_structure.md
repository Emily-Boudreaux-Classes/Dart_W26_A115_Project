# Directory Structure and Rules

Here we provide a breakdown of the naming schemes for branches, directories in the repository, and other things within the code. 

## Directories and Descriptions:
- **`/modules`**
    - contains everything related to the project. This includes code to make analysis plots, run stellar structure code, etc.
- **`/cache`**
    - this directory will contain anything that is not meant for the user, but must still be saved for I/O purposes
- **`/run`**
    - contains everything needed for the user to generate plots and run the stellar simulation code (maybe in a `/run/simulate` and `/run/analysis` directory?)
- **`/output`**
    - `/plots`
        - plots from analysis
    - `/models`
        - the actual model tables..?

## Branch Naming schemes:
 - `feat/MODULE` = Feature being added
 - `test/MODULE` = Test being done in the first week 
 - `debug/feat/MODULE1` = Debugging a feature


