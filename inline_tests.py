import os

out_dir = r'c:\Workspace\Projects\DeepLearning\ADSA--simplified\CNN_Extractor_&_Attack_simulation\jup_notebooks'
test_file = os.path.join(out_dir, 'stage3_tests.py')

with open(test_file, 'r', encoding='utf-8') as f:
    test_content = f.read()

# Remove the imports from test_content as they are already in the main scripts
# Just extract the function
func_start = test_content.find('def run_forensic_diagnostics')
test_func = test_content[func_start:]

scripts = ['stage3_phase0_verify.py', 'stage3_train_phase1.py', 'stage3_train_phase2.py', 'stage3_train_phase3.py', 'stage3_train_phase4.py']

for script in scripts:
    path = os.path.join(out_dir, script)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'from stage3_tests import *' in content:
        content = content.replace('from stage3_tests import *', test_func)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Inlined tests into {script}")
    else:
        print(f"Already inlined or not found in {script}")

