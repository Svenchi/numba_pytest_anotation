# Custom anotations for testing numba
An issue with numba library that compiles Python code into machine code is that getting coverage results of test runs is impossible. Unitest currently does not return coverage of numba code.
To alleviate this issue I created a custom annotation that detects if the code is run in test mode by checking if the "unittest" module is present, if it is, the numba code is compiled in python mode which gives us valid coverage.

# Proof that it works
As a proof test directory contains two same python scripts located in numba_function_custom_anotation.py and numba_function_default_anotation.py that only differ in the annotations used. the default_annotation functions use default numba annotations, while custom annotations use previously explained custom logic.
Two identical test suits have been setup in the test directory for these different functions.

Running tests locally produces the following coverage for the methods with differing annotations:
```console
test/test_functions/numba_function_custom_anotation.py       23      2    91%
test/test_functions/numba_function_default_anotation.py      23     15    35%
```
As seen in these results, coverage is much more representative with the custom annotation.


## executing proof:
to setup poetry environment run:
```console
poetry install
```

generate coverage file:
```console
poetry run coverage run -m pytest
```

get coverage details:
```console
poetry run coverage report
```

After getting coverage details, the same result should appear as was shown before.