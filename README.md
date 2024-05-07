# pershin_algorithms_wiki
Тут будут собраны пояснения и материалы к курсу Антона Юрьевича Першина по алгоритмам. 

## Getting started

Set up your environment

### VSCode

Go to `Run and Debug` in the left panel, create a new launch file, select `Python File` and add the following field:
```yaml
"env": {
    "PYTHONPATH": "${workspaceFolder}${pathSeparator}${env:PYTHONPATH}"
}
```

### MacOs (homebrew or python.app)
Go to `Terminal.app` and paste this command:
```bash
export PYTHONPATH=/Users/path_to_your_github_clone_directory:$PYTHONPATH
```
### MacOs (anaconda3 or miniconda)
Go to `Terminal.app` and paste this command:
```bash
conda install conda-build
conda develop /Users/path_to_your_github_clone_directory
```
Then you can check you PYTHONPATH with this command
```bash
python -c "import sys; print('\n'.join(sys.path))"
```
*Note: python inside your console and inside your PyCharm/Jupyter/DataSpell may be different so run commands in terminal of this app*

Another way to check your PYTHONPATH (most accurate)
paste this code (first line) in some .py file (for example: practicum_2/dfs.py ) and run the file
```python
print(sys.path) # you should see /Users/path_to_your_github_clone_directory
```
