import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import copy
import numpy as np
from tqdm import tqdm
import time
import json
from openai import OpenAIError,OpenAI
import backoff
import traceback

from llm_test.llm_module import Agent, API_KEY_R17B, API_KEY_SIQI, API_URL, API_URL_R17B, MODEL_SELECTION

from types import SimpleNamespace

class OraclePlanner(object):
    def __init__(
            self,
            environment_fn,
            agent_fn,
            args,
            run_predefined_actions=False,
            oracle_prompt_path=None,
        ):
        self.env_fn = environment_fn
        self.agents = agent_fn
        self.args = args
        self.num_agents = len(agent_fn)
        self.task_goal = None

        # define some args for the oracle planner
        self.oracle_prompt_path = oracle_prompt_path

        # define parameters that were used but not initialized
        self.source = args.source
        self.lm_id = args.lm_id
        self.device = None
        self.chat = True
        self.debug = args.debug

        # define the llm engine and generator for the oracle agent
        if self.source == 'openai':
            api_key = args.api_key # your openai api key
            organization=args.organization # your openai organization

            client = OpenAI(api_key = api_key, organization=organization)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": args.top_p,
                    "n": args.n
                }
        elif self.source == 'llm_module':

            api_key = API_KEY_SIQI
            api_url = API_URL
            model = MODEL_SELECTION
            # model = "gpt-4o-2024-11-20"

            client = Agent(model=model, api_url=api_url, api_key=api_key)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": 1.0,
                    "n": args.n
                }
            # self.device = args.device
            # self.lm_id = args.lm_id
            # self.chat = True
            # self.llm_module = Agent(model=self.lm_id, device=self.device)
            # self.sampling_params['model'] = self.lm_id

        def lm_engine(source, lm_id, device):

            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            prompt.insert(0,{"role":"system", "content":"You are a helper assistant."})
                            response = client.chat.completions.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            if self.debug:
                                with open(f"./chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response.choices[i].message.content for i in
                                                    range(sampling_params['n'])]
                            if 'gpt-4-0125-preview' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.01 / 1000 + response.usage.completion_tokens * 0.03 / 1000
                            elif 'gpt-3.5-turbo-1106' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == 'llm_module':
                    try:
                        if self.chat:
                            prompt.insert(0,{"role":"system", "content":"You are a helper assistant."})
                            response = client.respond_once_all_args(
                                messages=prompt, **sampling_params
                            )
                            # response = SimpleNamespace(**response)
                            if self.debug:
                                with open(f"./chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['message']['content'] for i in range(sampling_params['n'])]
                            if 'gpt-4-0125-preview' in self.lm_id or 'gpt-4o-2024-11-20' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.01 / 1000 + response['usage']['completion_tokens'] * 0.03 / 1000
                            elif 'gpt-3.5-turbo-1106' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except Exception as e:
                        print(e)
                        raise e

                else:
                    raise ValueError("invalid source")
                return generated_samples, usage 
            
            return _generate

        self.generator = lm_engine(self.source, self.lm_id, self.device)

    def get_oracle_prompt(self, obs_text, goal_instruction, num_agents, dialogue_history):
        '''
        Get the oracle prompt
        '''
        with open(self.oracle_prompt_path, 'r') as f:
            oracle_prompt = f.read()
        oracle_prompt = oracle_prompt.replace('#AGENT_OBSERVATIONS#', obs_text)
        oracle_prompt = oracle_prompt.replace('#TASK_GOAL#', goal_instruction)
        oracle_prompt = oracle_prompt.replace('#NUMBER_AGENTS#', str(num_agents))
        oracle_prompt = oracle_prompt.replace('#DIALOGUE_HISTORY#', dialogue_history)
        print(dialogue_history)
        
        return oracle_prompt
    
    def oracle_planning_vanilla(
            self,
            obs_text,
            goal_instruction,
            num_agents,
            dialogue_history,
        ):
        '''
        Doing vanilla oracle planning, which is the process of the oracle agent generating the plan for the task.
        Args:
            obs_text: str, the text of the observation of the agents
            goal_instruction: str, the goal instruction of the task
            num_agents: int, the number of agents
            dialogue_history: str, the dialogue history
        Returns:
            message: str, the message of the oracle agent
            usage: int, the usage of the llm engine
        '''
        # import the structured prompt and insert the information into the prompt
        oracle_prompt = self.get_oracle_prompt(obs_text, goal_instruction, num_agents, dialogue_history)
      
        chat_prompt = [{"role": "user", "content": oracle_prompt}]
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        message = outputs[0]

        return message, usage

    def extract_structured_message(self, message):
        '''
        Extract the structured message from the message from the vanilla planning of the oracle agent
        '''
        extract_prompt = message + '\n' + \
            'Extract from the above paragraph the content of the format "Hello <class name>(id): message.". ' + \
            'Then output the contents of this section. Be careful not to output any superfluous content, exactly in the format given. ' + \
            'If the above paragraph is not exactly formatted as "Hello <class name>(id): #message#.", output similar content in this format. ' + \
            'As an example, the output might read: "Hello <robot dog>(0): please movetowards the <door>(1), and then open the <door>(1)". ' + \
            'If this format does not appear in the preceding text, please summarize the above content into this format for output. ' + \
            'To emphasize once again, the names of all objects and agent robots must be enclosed in <>, and the (id) must not be omitted. ' + \
            'Class name missing <> and (id) should be completed with these elements. ' + \
            'Please strictly follow this format in the output content.' 
        
        chat_prompt = [{"role": "user", "content": extract_prompt}]
        outputs, usage = self.generator(chat_prompt , self.sampling_params)
        message_output = outputs[0]
        
        return message_output, usage

