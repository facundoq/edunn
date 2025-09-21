
We are open to contributions! Specifically:

1. Improvement to current guides.
2. Translation of guides to other languages.
3. Additional guides.
4. Additional layers/models/optimizers (with the corresponding guide).


## :bulb: Project organization
A **release** of the library is a copy of the library without the key code that learners must implement.

To generate a **release**, EduNN's reference implementation in folder `edunn` is stripped of `"""YOUR IMPLEMENTATION START"""` and `"""YOUR IMPLEMENTATION END"""` comment pairs, and all the code in between. The resulting code is stored in the `generated` folder.
 
Guides can be written for different languages (english, spanish, etc), and are zipped with a release to generate a downloadable guide. 


# :hammer: How to contribute

1. Clone the repository
2. Create a venv, install packages in `requirements.txt` and `requirements_dev.txt`
2. Run `edit_guides.sh` to fire up a jupyter-notebook server that can import `edunn` correctly (recommended to avoid path issues, but you can do this in other ways if you are knowledgeable in python)
2. Add/modify guide and/or a `edunn` component
3. Generate a release for guides and/or `edunn`  (see next steps)

##  :package: New `edunn` releases 

If you modified the `edunn` library itself, then:

1. If necessary, add/update tests in the `test` folder
2. Verify all tests are still working by running `pytest` in the root folder of the project  
3. Generate a new release (only if you modified `edunn`) by running`export_code.py`. This will copy all code in `edunn` to `generated` and remove the reference implementation.
4. Recreate guides (see next section)
5. Create pull request
6.  Update pipy (maintainers only)

   
## :notebook: New guide releases

If you modified a guide of language `<lang>`:

1. Export guide (if modified) with `zip_guide.py <lang>` (ie, `zip_guide.py es` for spanish)
2. Check exported zip works correctly
3. Create pull request 
4. Add release to github (maintainers only)

