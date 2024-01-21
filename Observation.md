### File system of components:
- In the "code" argument of the "command" function, the directory specified will be copied to the registered component under "code". Hence, if the module that the component (e.g., that is used by "train.py") belong to that directory, it is copied and the component code can run fine. Else it will cause ModuleNotFound error.
- Example:
    - This will work: the whole root folder that contains train.py is copied to online component
        ```bash
            .
            ├── prep_data.py
            ├── train.py        #uses utilities.py
            └── dp100/
                ├── __init__.py
                └── utilities.py
        ```

    - This will work: just a more efficient version of the above, so that the other content in the root folder are not copied to the component, wasting time, storage and potentially exposing unnecessary elements of the codebase online
        ```bash
            .
            └── dp100/
                ├── __init__.py
                ├── train.py
                ├── prep_data.py
                └── src/
                    └── utilities.py
        ```
    
    - This will not work: only the "components" folder is copied over
        ```bash
            .
            ├── components/
            │   ├── prep_data.py
            │   └── train.py
            └── dp100/
                ├── __init__.py
                └── utilities.py
        ```
