
We are open to contributions! Specifically:

1. Improvement to current guides.
2. Translation of guides to other languages.
3. Additional guides.
4. Additional layers/models/optimizers (with the corresponding guide).


# :bulb: Project organization
A **release** of the library is a copy of the library without the key code that learners must implement.

To generate a **release**, SimpleNN's reference implementation in folder `simplenn` is stripped of `### YOUR IMPLEMENTATION START  ###` and `### YOUR IMPLEMENTATION END  ###` comment pairs, and all the code in between. The resulting code is stored in the `generated` folder.
 
Guides can be written for different languages (english, spanish, etc), and can be zipped with a release to generate a downloadable guide. 


# :hammer: How to contribute

1. Clone the repository
2. Run `notebook_guides.sh` to fire up a jupyter-notebook server that can import `simplenn` correctly (recommended to avoid path issues, but you can do this in other ways if you are knowledgeable in python)
2. Add/modify guide and/or `simplenn` component
3.

#  :package: New `simplenn` releases 

If you modified the `simplenn` library itself, then:

1. If necessary, add/update tests in the `test` folder
2. Verify all tests in `test` are still working  
3. Generate a new release (only if you modified `simplenn`) by running`export_code.py`. This will copy all code in `simplenn` to `generated` and remove the reference implementation.
4. Recreate guides (see next section)
5. Create pull request
6. (maintainer only) Update pipy

   
# :notebook: New guide releases

If you modified a guide:

1. Export guide (if modified) with `zip_guide.sh`
2. Check exported zip works correctly
3. Create pull request 
4. Add release (maintainer only)

