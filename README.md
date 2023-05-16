# TransLEARNer: NLP-Aided RL


To install requirements:
```
pip install -r requirements.txt
```

Additionally, make sure to install the requirements of the modified Multimodal-Toolkit (multimodal-transformers):
```
cd Multimodal-Toolkit && pip install .
```

To train RL agent:
```
python main.py --task 2

options:
  -h, --help            show this help message and exit
  --task TASK           task number 0-6{0: <class 'language.tasks.DownLadderJumpRight'>, 1: <class 'language.tasks.ClimbDownRightLadder'>, 2: <class
                        'language.tasks.JumpSkullReachLadder'>, 3: <class 'language.tasks.JumpSkullGetKey'>, 4: <class
                        'language.tasks.ClimbLadderGetKey'>, 5: <class 'language.tasks.ClimbDownGoRightClimbUp'>, 6: <class
                        'language.tasks.JumpMiddleClimbReachLeftDoor'>}
  --lang_rewards LANG_REWARDS
                        'true' = use language rewards (default)
                        'false' (or anything else = don't use language rewards
  --lang_reward_interval LANG_REWARD_INTERVAL
                        interval for reward calculation (default = 10)
  --timesteps TIMESTEPS
                        number of timesteps to play
  --render RENDER       render mode - 'true' / 'false' (defaults to 'false')
  --instr INSTR         provide a list of instructions separated by "[SEP]" (ex. "go left and jump [SEP] jump left [SEP] left jump")
  --lang_coef LANG_COEF
                        language reward coefficient (language rewards are multiplief by the coefficient)
                        default = 0.05
  --save_folder SAVE_FOLDER
                        save path for task logs
  --device DEVICE       device
  
  
  
```

NOTE: optimal language reward coefficient will vary by task. If the amount of language rewards starts to grow at a fast pace and and the external rewards flatten out, the coefficient is too high.

To train TransLEARNer model:
You'll need to push it to the Huggingface hub, so log in with your token before you do this.

Log in:
```
pip install huggingface_hub
git config --global credential.helper store
huggingface-cli login
```

Ready to train:
```
python language/multimodal_model.py --save_repo "your_hf_username/repo_name"
```
