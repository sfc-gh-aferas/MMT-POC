x rename repo
x remove all references to CCCI
x flatten repo for ease of config reference
- setup.sql
- config
    - data configs
        - train source table name
        - test source table name
        - partition key
        - time key
    - Include data generation?
        - source table definition
        - granuality
        - # partitions and timesteps
        - data splitting strategy
        - start date
    - Train config
        - Compute pool type
        - # nodes
    - Infer config
        - Compute pool type
        - # nodes
- prompt to update setup.sql if needed
- add model registration to train
- model catalog should save metrics as variant for flexibility
- use registered model in infer
- update readme with instructions
    - train and infer funcs MUST be edited
    - BYOD or generation approach and requirements for each
    - note that does not include role and privilege management

