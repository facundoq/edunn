#!/usr/bin/env bash
echo "***********************************************************"
echo "*** Running 'jupyter notebook' with base dir of project ***"
echo "*** added to path so that guides can import edunn       ***"
echo "*** Use _only_ for development; exported guides should  ***"
echo "*** have _their_ copy of edunn in their base dir.       ***"
echo "***********************************************************"

this="$(dirname "$(realpath "$0")")";
PYTHONPATH="$this" jupyter notebook