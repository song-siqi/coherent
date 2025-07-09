import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import json
import copy
import backoff
import openai
from openai import OpenAIError, OpenAI
from types import SimpleNamespace

from llm_test.llm_module import Agent, API_KEY_R17B, API_KEY_SIQI, API_URL, API_URL_R17B, MODEL_SELECTION

# inherited from Coherent
class LLM:
	def __init__(self, source, lm_id, args):

		self.args = args
		self.debug = args.debug
		self.source = args.source
		self.lm_id = args.lm_id
		self.chat = True
		self.total_cost = 0
		self.device = None
		self.record_dir = f'./log/{args.env}.txt'

		if self.source == 'openai':

			api_key = args.api_key  # your openai api key
			organization= args.organization # your openai organization

			client = OpenAI(api_key = api_key, organization=organization)
			if self.chat:
				self.sampling_params = {
					"max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": 1.0,
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
			# @backoff.on_exception(backoff.expo, OpenAIError)
			@backoff.on_exception(backoff.expo, Exception)
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

	def parse_answer(self, available_actions, text):
		
		text = text.replace("_", " ")
		text = text.replace("takeoff from", "takeoff_from")
		text = text.replace("land on", "land_on")

		for i in range(len(available_actions)):
			action = available_actions[i]
			if action in text:
				return action
		self.write_log_to_file('\nThe first action parsing failed!!!')

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
				return action
		self.write_log_to_file('\nThe second action parsing failed!!!')

		print("WARNING! No available action parsed!!! Output plan NONE!\n")
		return None

	def get_available_plans(
			self,
			agent_node,
			next_rooms,
			all_landable_surfaces,
			landable_surfaces,
			on_surfaces,
			grabbed_objects,
			reached_objects,
			unreached_objecs,
			on_same_surface_objects
		):
		"""
		'quadrotor':
		[land_on] <surface>
		[movetowards] <surface>/<next_room>
		[takeoff_from] <surface>

		'robot dog':
		[open] <container>/<door>
		[close] <container>/<door>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>
		[movetowards] <object>

		'robot arm':
		[open] <container>
		[close] <container>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>

		"""
		available_plans = []
		if agent_node["class_name"] == "quadrotor":
			other_landable_surfaces = []
			if "FLYING" in agent_node["states"]:
				if landable_surfaces is not None:
					available_plans.append(f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})")
					all_landable_surfaces.remove(landable_surfaces)
					other_landable_surfaces = copy.deepcopy(all_landable_surfaces)
				if len(other_landable_surfaces) != 0:
					for surface in other_landable_surfaces :
						available_plans.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
				for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

			if "LAND" in agent_node["states"]:
				if on_surfaces is not None:
					available_plans.append(f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})")

		if agent_node["class_name"] == "robot dog" or agent_node["class_name"] == "robot_dog":
			# if grabbed_objects is not None:
			# 	available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			# The robotic dog is not allowed to put things on the floor. If it needs to open the door and has something in its hand, it needs to find a low surface to put things first
			if len(reached_objects) != 0:
				for reached_object in reached_objects:
					if grabbed_objects is None:
						if 'CONTAINERS' in reached_object['properties'] and 'CLOSED' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'CLOSED' in reached_object['states']:
							available_plans.append(f"[open] <{reached_object['class_name']}>({reached_object['id']})")
						if 'CONTAINERS' in reached_object['properties'] and 'OPEN' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'OPEN' in reached_object['states']:
							available_plans.append(f"[close] <{reached_object['class_name']}>({reached_object['id']})")
						if 'GRABABLE' in reached_object['properties']:
							available_plans.append(f"[grab] <{reached_object['class_name']}>({reached_object['id']})")
					if grabbed_objects is not None:
						if 'CONTAINERS' in reached_object['properties'] and ('OPEN' in reached_object['states'] or "OPEN_FOREVER" in reached_object['states']):
							available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{reached_object['class_name']}>({reached_object['id']})")
						if 'SURFACES' in reached_object['properties']:
							available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{reached_object['class_name']}>({reached_object['id']})")
			
			if len(unreached_objecs) != 0:
				for unreached_object in unreached_objecs:
					available_plans.append(f"[movetowards] <{unreached_object['class_name']}>({unreached_object['id']})")
			for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")


		if agent_node['class_name'] == 'robot arm' or agent_node['class_name'] == 'robot_arm':
			if grabbed_objects is not None:
				available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			for on_same_surface_object in on_same_surface_objects:
				if grabbed_objects is None:
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'OPEN' in on_same_surface_object['states']:
						available_plans.append(f"[close] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'CLOSED' in on_same_surface_object['states']:
						available_plans.append(f"[open] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'GRABABLE' in on_same_surface_object['properties']:
						available_plans.append(f"[grab] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

				if grabbed_objects is not None:
					
					if 'CONTAINERS' in on_same_surface_object['properties'] and ('OPEN' in on_same_surface_object['states'] or "OPEN_FOREVER" in on_same_surface_object['states']):
						available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'SURFACES' in on_same_surface_object['properties']:
						available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"
		print(agent_node["class_name"],agent_node['id'])
		print(available_plans)
		return plans, len(available_plans), available_plans

		
	def run(self, agent_node, chat_agent_info, current_room, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects):
		info = {"num_available_actions": None,
			"prompts": None,
			"outputs": None,
			"plan": None,
			"action_list": None,
			"cost":self.total_cost, 
			f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost}

		prompt_path = chat_agent_info['prompt_path']
		with open(prompt_path, 'r') as f:
			agent_prompt = f.read()

		available_plans, num, available_plans_list = \
			self.get_available_plans(
				agent_node,
				next_rooms,
				all_landable_surfaces,
				landable_surfaces,
				on_surfaces,
				grabbed_objects,
				reached_objects,
				unreached_objecs,
				on_same_surface_objects,
			)
		
		agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
		agent_prompt = agent_prompt.replace('#ACTIONLIST#', available_plans)
		agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
		
		if self.debug:
			print(f"cot_prompt:\n{agent_prompt}")
		chat_prompt = [{"role": "user", "content": agent_prompt}]
		outputs, usage = self.generator(chat_prompt, self.sampling_params)
		output = outputs[0]

		self.write_log_to_file(output+'\n111111111')
		self.total_cost += usage
		info['cot_outputs'] = outputs

		if self.debug:
			print(f"cot_output:\n{output}")
			print(f"total cost: {self.total_cost}")
		sentences = output.split(".")
		first_sentence = sentences[0].upper()
		print("#" *20)
		print("the first sentence is", first_sentence)
		print("#" *20)

		if first_sentence == "YES I CAN":
			chat_prompt = [{"role": "user", "content": agent_prompt},
							{"role": "assistant", "content": output},
							{"role": "user", "content": "Answer with only one best next action in the list of available actions. So the answer is"}]

			outputs, usage = self.generator(chat_prompt, self.sampling_params)
			output = outputs[0]
			self.total_cost += usage
			self.write_log_to_file(output+'\n2222222222222')
			sentences = output.split(".")
			first_sentence = sentences[0].upper()
			if first_sentence != "SORRY I CANNOT": 

				if self.debug:
					print(f"cot_output:\n{output}")
					print(f"total cost: {self.total_cost}")

				plan = self.parse_answer(available_plans_list, output)
				if plan is None:
					plan_str = 'no plan'
				else:
					plan_str = plan
				print(plan)
				if self.debug:
					print(f"plan: {plan}\n")
				info.update({"num_available_actions": num,
						"prompts": chat_prompt,
						# "outputs": outputs,
						"plan": plan,
						"action_list": available_plans_list,
						f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost})
				message = f" The action I finally decided to perform is {plan_str}. "

				prompt_path = self.args.judge_prompt_path
				with open(prompt_path, 'r') as f:
					prompt = f.read()
				prompt = prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
				prompt = prompt.replace('#PLAN#', plan_str)
				prompt = prompt.replace('#AGENT#', f"<{agent_node['class_name']}>")
				prompt = [{"role": "user", "content": prompt}]
				outputs, usage = self.generator(prompt, self.sampling_params)
				output = outputs[0]
				self.total_cost += usage
				message += output
				self.write_log_to_file(output+'\n333333333333333333')
				info.update({"outputs": message})


		if first_sentence == "SORRY I CANNOT":
			output = output[16].lower() + output[17:]
			message = f"Sorry, the current actions I can perform cannot complete this instrcution. Possible reasons would be {output} My current actionlist is: {available_plans}"
			self.write_log_to_file(message+'\n4444444444444')
		info['cost'] = self.total_cost	
		self.write_log_to_file(f"total cost: {self.total_cost}")
		info.update({"outputs": message})
		return message, info

	def write_log_to_file(self,log_message, file_name=None):
		file_name = self.record_dir
		with open(file_name, 'a') as file:  
			file.write(log_message + '\n')  

# inherited from Coherent
class LLM_agent:
	"""
	LLM agent class
	"""
	def __init__(self, agent_id, args, agent_node, init_graph):

		self.agent_node = agent_node
		self.agent_id = agent_id
		self.init_graph = init_graph
		self.init_id2node = {x['id']: x for x in init_graph['nodes']}
		self.source = args.source
		self.lm_id = args.lm_id
		self.args = args
		self.LLM = LLM(self.source, self.lm_id, self.args)
		self.unsatisfied = {}
		self.steps = 0
		self.plan = None
		self.current_room = None
		self.grabbed_objects = None
		self.goal_location = None
		self.goal_location_id = None
		self.last_action = None
		self.id2node = {}
		self.id_inside_room = {}
		self.satisfied = []
		self.reachable_objects = []


	def LLM_plan(self):

		return self.LLM.run(self.agent_node, self.chat_agent_info, self.current_room, self.next_rooms, self.all_landable_surfaces,self.landable_surfaces, 
					  self.on_surfaces, self.grabbed_objects, self.reachable_objects, self.unreached_objects, self.on_same_surfaces)


	def check_progress(self, state, goal_spec):
		unsatisfied = {}
		satisfied = []
		id2node = {node['id']: node for node in state['nodes']}

		for key, value in goal_spec.items():
			elements = key.split('_')
			self.goal_location_id = int((re.findall(r'\((.*?)\)', elements[-1]))[0])
			self.target_object_id = int((re.findall(r'\((.*?)\)', elements[1]))[0])
			cnt = value[0]
			for edge in state['edges']:
				if cnt == 0:
					break
				if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and edge['from_id'] == self.target_object_id:
					satisfied.append(id2node[edge['from_id']])  # A list of nodes that meet the goal
					cnt -= 1
					# if self.debug:
					# 	print(satisfied)
			if cnt > 0:
				unsatisfied[key] = value  
		return satisfied, unsatisfied


	def get_action(self, observation, chat_agent_info, goal):

		satisfied, unsatisfied = self.check_progress(observation, goal) 
		# print(f"satisfied: {satisfied}")
		if len(satisfied) > 0:
			self.unsatisfied = unsatisfied
			self.satisfied = satisfied

		obs = observation
		self.grabbed_objects = None
		self.reachable_objects = []
		self.landable_surfaces = None
		self.on_surfaces = None
		self.all_landable_surfaces = []
		self.all_landable_surfaces = [x for x in obs['nodes'] if 'LANDABLE' in x['properties']]
		self.on_same_surfaces = []
		self.on_same_surfaces_ids = []
		self.chat_agent_info = chat_agent_info

		self.id2node = {x['id']: x for x in obs['nodes']}

		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			
			if x == self.agent_node['id']:

				if r == 'INSIDE':
					self.current_room = self.id2node[y]
				if r == 'ON' :
					self.on_surfaces = self.id2node[y]
					if self.agent_node['class_name'] == 'robot arm' or self.agent_node['class_name'] == 'robot_arm':
						for i in range(3):
							for edge in obs['edges']:
								if (edge['from_id'] != x and edge['to_id'] == y and edge['relation_type'] == 'ON') or (edge['from_id'] != x and edge['to_id'] in self.on_same_surfaces_ids and edge['relation_type'] == 'ON') or (edge['from_id'] != x and edge['to_id'] in self.on_same_surfaces_ids and edge['relation_type'] == 'INSIDE') :
									self.on_same_surfaces_ids.append(edge['from_id'])
									#self.on_same_surfaces.append(self.id2node[edge['from_id']])  # Find any contain or surface on the table
									if 'SURFACES' in self.id2node[edge['from_id']]['properties'] or 'CONTAINERS' in self.id2node[edge['from_id']]['properties']:
										for ee in obs['edges']:
											if ee['to_id'] == edge['from_id'] and (ee['relation_type'] == 'INSIDE' or ee['relation_type'] == 'ON'):
												self.on_same_surfaces_ids.append(ee['from_id'])
												#self.on_same_surfaces.append(self.id2node[ee['from_id']]) # The goal here is to find objects that are not directly on the surface
								self.on_same_surfaces_ids = list(set(self.on_same_surfaces_ids))	
						for id in self.on_same_surfaces_ids:
							self.on_same_surfaces.append(self.id2node[id])
				
				if r == 'HOLD':
					# self.grabbed_objects.append(y)
					self.grabbed_objects = self.id2node[y]
				if r == 'CLOSE':
					self.reachable_objects.append(self.id2node[y])
				if r == 'ABOVE' and 'LANDABLE' in self.id2node[y]['properties']:
					self.landable_surfaces = self.id2node[y]

		self.unreached_objects = copy.deepcopy(obs['nodes'])
		for node in obs['nodes']:
			if node == self.grabbed_objects or node in self.reachable_objects:
				self.unreached_objects.remove(node)
			elif node['category'] == 'Rooms' or node['category'] == 'Agents' or node['category'] == 'Floor' or "HIGH_HEIGHT" in node['properties'] or 'ON_HIGH_SURFACE' in node['properties']:
				self.unreached_objects.remove(node)
		# NOTE: The idea here is to find the places that the robotic dog has not reached, 
		# 		remove what it already has in its hand, remove what it is close to, remove the room, 
		# 		the floor, the agent itself, the high surface and what is on the high surface

		self.doors = []
		self.next_rooms = []
		self.doors = [x for x in obs['nodes'] if x['class_name'] == 'door']
		for door in self.doors:
			for edge in self.init_graph['edges']:
				if edge['relation_type'] == "LEADING TO" and edge['from_id'] == door['id'] and edge['to_id'] != self.current_room["id"]:
						self.next_rooms.append([self.init_id2node[edge['to_id']], door])

		info = {'graph': obs,
				"obs": {	
						 "agent_class": self.agent_node["class_name"],
						 "agent_id":self.agent_node["id"],
						 "grabbed_objects": self.grabbed_objects,
						 "reachable_objects": self.reachable_objects,
						 "on_surfaces": self.on_surfaces,
						 "landable_surfaces": self.landable_surfaces,
						 "doors": self.doors,
						 "next_rooms": self.next_rooms,
						 "objects_on_the_same_surfaces": self.on_same_surfaces,
						 "satisfied": self.satisfied,
						 "current_room": self.current_room['class_name'],
						},
				}

		message, a_info = self.LLM_plan()
		if a_info['plan'] is None: 
			print("No more things to do!")
		plan = a_info['plan']
		a_info.update({"steps": self.steps})
		info.update({"LLM": a_info})

		return plan, message, info


class FeedbackAgent:
    def __init__(self, agent_id, args, agent_node, init_graph):
        # LLM parameters
        self.args = args
        self.debug = args.debug
        self.source = args.source
        self.lm_id = args.lm_id
        self.chat = True
        self.total_cost = 0
        self.device = None
        self.record_dir = f'./log/{args.env}.txt'

        # LLM_agent parameters
        self.agent_node = agent_node
        self.agent_id = agent_id
        self.init_graph = init_graph
        self.init_id2node = {x['id']: x for x in init_graph['nodes']}
        self.id2node = None
        self.current_room = None
        self.grabbed_objects = None
        self.reachable_objects = []
        self.on_surfaces = None
        self.landable_surfaces = None
        self.on_same_surfaces = []
        self.satisfied = []
        self.steps = 0
        self.unreached_objects = []
        self.doors = []
        self.next_rooms = []
        self.all_landable_surfaces = []
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id_inside_room = {}
        self.unsatisfied = {}

        # Initialize LLM client
        if self.source == 'openai':
            api_key = args.api_key
            organization = args.organization
            self.client = OpenAI(api_key=api_key, organization=organization)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": args.max_tokens,
                    "temperature": args.t,
                    "n": args.n
                }

        elif self.source == 'llm_module':
            api_key = API_KEY_SIQI
            api_url = API_URL
            model = MODEL_SELECTION
            self.client = Agent(model=model, api_url=api_url, api_key=api_key)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": args.max_tokens,
                    "temperature": args.t,
                    "n": args.n
                }

    def check_progress(self, state, goal_spec):
        """Check progress towards goal completion"""
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            self.goal_location_id = int((re.findall(r'\((.*?)\)', elements[-1]))[0])
            self.target_object_id = int((re.findall(r'\((.*?)\)', elements[1]))[0])
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and edge['from_id'] == self.target_object_id:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
            if cnt > 0:
                unsatisfied[key] = value

        return satisfied, unsatisfied

    def get_action(self, observation, chat_agent_info, goal):
        """Get next action based on current observation and goal"""
        # Reset all state variables to prevent any state leakage between calls
        self._reset_state()
        
        # Update goal progress
        satisfied, unsatisfied = self.check_progress(observation, goal)
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied

        obs = observation
        self.chat_agent_info = chat_agent_info
        self.id2node = {x['id']: x for x in obs['nodes']}
        
        # Update all environment state information
        self._update_environment_state(obs)

        # Get plan and message
        message, a_info = self.LLM_plan(chat_agent_info)
        # import ipdb; ipdb.set_trace()
        if a_info['plan'] is None:
            print("No more things to do!")
        plan = a_info['plan']
        self.last_action = plan
        a_info.update({"steps": self.steps})
        # self.steps += 1

        # Prepare observation info
        info = {
            'graph': obs,
            "obs": {
                "agent_class": self.agent_node["class_name"],
                "agent_id": self.agent_node["id"],
                "grabbed_objects": self.grabbed_objects,
                "reachable_objects": self.reachable_objects,
                "on_surfaces": self.on_surfaces,
                "landable_surfaces": self.landable_surfaces,
                "doors": self.doors,
                "next_rooms": self.next_rooms,
                "objects_on_the_same_surfaces": self.on_same_surfaces,
                "satisfied": self.satisfied,
                "current_room": self.current_room['class_name'],
            },
            "LLM": a_info
        }

        return plan, message, info

    def _reset_state(self):
        """Reset all state variables to prevent state leakage between calls"""
        self.grabbed_objects = None
        self.reachable_objects = []
        self.landable_surfaces = None
        self.on_surfaces = None
        self.all_landable_surfaces = []
        self.on_same_surfaces = []
        self.on_same_surfaces_ids = []
        self.current_room = None
        self.doors = []
        self.next_rooms = []
        self.unreached_objects = []
        # Do not reset these as they need to persist:
        # self.satisfied
        # self.unsatisfied
        # self.steps
        # self.last_action
        # self.goal_location
        # self.goal_location_id
        # self.id_inside_room

    def _update_environment_state(self, obs):
        """Update all environment state based on current observation"""
        # Update landable surfaces
        self.all_landable_surfaces = [x for x in obs['nodes'] if 'LANDABLE' in x['properties']]
        
        # Process edges to update object relationships
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            
            if x == self.agent_node['id']:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y]
                if r == 'ON':
                    self.on_surfaces = self.id2node[y]
                    if self.agent_node['class_name'] in ['robot arm', 'robot_arm']:
                        self._update_reachable_objects_on_surface(obs, y)
                if r == 'HOLD':
                    self.grabbed_objects = self.id2node[y]
                if r == 'CLOSE':
                    self.reachable_objects.append(self.id2node[y])
                if r == 'ABOVE' and 'LANDABLE' in self.id2node[y]['properties']:
                    self.landable_surfaces = self.id2node[y]

        # Update unreached objects
        self.unreached_objects = copy.deepcopy(obs['nodes'])
        for node in obs['nodes']:
            if node == self.grabbed_objects or node in self.reachable_objects:
                self.unreached_objects.remove(node)
            elif node['category'] in ['Rooms', 'Agents', 'Floor'] or "HIGH_HEIGHT" in node['properties'] or 'ON_HIGH_SURFACE' in node['properties']:
                self.unreached_objects.remove(node)
        # NOTE: The idea here is to find the places that the robotic dog has not reached, 
        #       remove what it already has in its hand, remove what it is close to, remove the room, 
        #       the floor, the agent itself, the high surface and what is on the high surface

        # Update doors and next rooms
        self.doors = [x for x in obs['nodes'] if x['class_name'] == 'door']
        self.next_rooms = []
        for door in self.doors:
            for edge in self.init_graph['edges']:
                if edge['relation_type'] == "LEADING TO" and edge['from_id'] == door['id'] and edge['to_id'] != self.current_room["id"]:
                    self.next_rooms.append([self.init_id2node[edge['to_id']], door])

    def _update_reachable_objects_on_surface(self, obs, surface_id):
        """Update the list of objects that the robot arm can reach on the same surface.
        
        This method identifies objects that are:
        1. Directly on the same surface
        2. Inside containers on the surface
        3. On other surfaces that are on the main surface
        
        Args:
            obs: The current observation containing all environment information
            surface_id: The ID of the surface the robot arm is currently on
        """
        on_same_surfaces_ids = []
        # Check up to 3 levels deep for nested objects
        for i in range(3):
            for edge in obs['edges']:
                # Check for objects on the same surface or inside containers
                if ((edge['from_id'] != self.agent_node['id'] and 
                     edge['to_id'] == surface_id and 
                     edge['relation_type'] == 'ON') or 
                    (edge['from_id'] != self.agent_node['id'] and 
                     edge['to_id'] in on_same_surfaces_ids and 
                     edge['relation_type'] in ['ON', 'INSIDE'])):
                    
                    on_same_surfaces_ids.append(edge['from_id'])
                    # If we find a surface or container, check for objects on/in it
                    if ('SURFACES' in self.id2node[edge['from_id']]['properties'] or 
                        'CONTAINERS' in self.id2node[edge['from_id']]['properties']):
                        for ee in obs['edges']:
                            if (ee['to_id'] == edge['from_id'] and 
                                ee['relation_type'] in ['INSIDE', 'ON']):
                                on_same_surfaces_ids.append(ee['from_id'])
        
        # Remove duplicates and update the list of reachable objects
        on_same_surfaces_ids = list(set(on_same_surfaces_ids))
        self.on_same_surfaces = [self.id2node[id] for id in on_same_surfaces_ids]

    def LLM_plan(self, chat_agent_info):
        """Implement LLM planning logic"""
        
        info = {
            "num_available_actions": None,
            "prompts": None,
            "outputs": None,
            "plan": None,
            "action_list": None,
            "cost": self.total_cost,
            f"<{self.agent_node['class_name']}>({self.agent_node['id']}) total_cost": self.total_cost
        }

        # Get available plans
        available_plans, num_plans, available_plans_list = self.get_available_plans()
        
        # Read and prepare prompt
        prompt_path = chat_agent_info['prompt_path']
        with open(prompt_path, 'r') as f:
            agent_prompt = f.read()

        agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
        agent_prompt = agent_prompt.replace('#ACTIONLIST#', available_plans)
        agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])

        if self.debug:
            print(f"cot_prompt:\n{agent_prompt}")

        # First LLM call to check if task is possible
        chat_prompt = [{"role": "user", "content": agent_prompt}]
        outputs, usage = self._generate(chat_prompt, self.sampling_params)
        output = outputs[0]
        self.write_log_to_file(output + '\n111111111')
        self.total_cost += usage
        info['cot_outputs'] = outputs

        if self.debug:
            print(f"cot_output:\n{output}")
            print(f"total cost: {self.total_cost}")

        sentences = output.split(".")
        first_sentence = sentences[0].upper()
        print("#" * 20)
        print("the first sentence is", first_sentence)
        print("#" * 20)

        message = ""
        if first_sentence == "YES I CAN":
            # Second LLM call to get specific action
            chat_prompt = [
                {"role": "user", "content": agent_prompt},
                {"role": "assistant", "content": output},
                {"role": "user", "content": "Answer with only one best next action in the list of available actions. So the answer is"}
            ]

            outputs, usage = self._generate(chat_prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            self.write_log_to_file(output + '\n2222222222222')

            sentences = output.split(".")
            first_sentence = sentences[0].upper()
            if first_sentence != "SORRY I CANNOT":
                if self.debug:
                    print(f"cot_output:\n{output}")
                    print(f"total cost: {self.total_cost}")

                plan = self.parse_answer(available_plans_list, output)
                plan_str = plan if plan is not None else 'no plan'
                print(plan)
                
                if self.debug:
                    print(f"plan: {plan}\n")
                
                info.update({
                    "num_available_actions": num_plans,
                    "prompts": chat_prompt,
                    "plan": plan,
                    "action_list": available_plans_list,
                    f"<{self.agent_node['class_name']}>({self.agent_node['id']}) total_cost": self.total_cost
                })
                
                message = f" The action I finally decided to perform is {plan_str}. "

                # Third LLM call for judgment/explanation
                prompt_path = self.args.judge_prompt_path
                with open(prompt_path, 'r') as f:
                    prompt = f.read()
                
                prompt = prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
                prompt = prompt.replace('#PLAN#', plan_str)
                prompt = prompt.replace('#AGENT#', f"<{self.agent_node['class_name']}>")
                prompt = [{"role": "user", "content": prompt}]
                
                outputs, usage = self._generate(prompt, self.sampling_params)
                output = outputs[0]
                self.total_cost += usage
                message += output
                self.write_log_to_file(output + '\n333333333333333333')

        elif first_sentence == "SORRY I CANNOT":
            output = output[16].lower() + output[17:]
            message = f"Sorry, the current actions I can perform cannot complete this instruction. Possible reasons would be {output} My current actionlist is: {available_plans}"
            self.write_log_to_file(message + '\n4444444444444')

        info['cost'] = self.total_cost
        info['outputs'] = message
        self.write_log_to_file(f"total cost: {self.total_cost}")
        
        return message, info

    def get_available_plans(self):
        '''
		'quadrotor':
		[land_on] <surface>
		[movetowards] <surface>/<next_room>
		[takeoff_from] <surface>

		'robot dog':
		[open] <container>/<door>
		[close] <container>/<door>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>
		[movetowards] <object>

		'robot arm':
		[open] <container>
		[close] <container>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>

		'''
        """Get available plans based on agent type and current state"""
        
        available_plans = []
        
        if self.agent_node["class_name"] == "quadrotor":
            other_landable_surfaces = []
            if "FLYING" in self.agent_node["states"]:
                if self.landable_surfaces is not None:
                    available_plans.append(f"[land_on] <{self.landable_surfaces['class_name']}>({self.landable_surfaces['id']})")
                    self.all_landable_surfaces.remove(self.landable_surfaces)
                    other_landable_surfaces = copy.deepcopy(self.all_landable_surfaces)
                if other_landable_surfaces:
                    for surface in other_landable_surfaces:
                        available_plans.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
                for next_room in self.next_rooms:
                    if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
                        available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

            if "LAND" in self.agent_node["states"] and self.on_surfaces is not None:
                available_plans.append(f"[takeoff_from] <{self.on_surfaces['class_name']}>({self.on_surfaces['id']})")

        elif self.agent_node["class_name"] in ["robot dog", "robot_dog"]:
            if self.reachable_objects:
                for reached_object in self.reachable_objects:
                    if self.grabbed_objects is None:
                        if ('CONTAINERS' in reached_object['properties'] and 'CLOSED' in reached_object['states']) or \
                           (reached_object['class_name'] == 'door' and 'CLOSED' in reached_object['states']):
                            available_plans.append(f"[open] <{reached_object['class_name']}>({reached_object['id']})")
                        if ('CONTAINERS' in reached_object['properties'] and 'OPEN' in reached_object['states']) or \
                           (reached_object['class_name'] == 'door' and 'OPEN' in reached_object['states']):
                            available_plans.append(f"[close] <{reached_object['class_name']}>({reached_object['id']})")
                        if 'GRABABLE' in reached_object['properties']:
                            available_plans.append(f"[grab] <{reached_object['class_name']}>({reached_object['id']})")
                    else:  # Has grabbed object
                        if 'CONTAINERS' in reached_object['properties'] and ('OPEN' in reached_object['states'] or "OPEN_FOREVER" in reached_object['states']):
                            available_plans.append(f"[putinto] <{self.grabbed_objects['class_name']}>({self.grabbed_objects['id']}) into <{reached_object['class_name']}>({reached_object['id']})")
                        if 'SURFACES' in reached_object['properties']:
                            available_plans.append(f"[puton] <{self.grabbed_objects['class_name']}>({self.grabbed_objects['id']}) on <{reached_object['class_name']}>({reached_object['id']})")

            if self.unreached_objects:
                for unreached_object in self.unreached_objects:
                    available_plans.append(f"[movetowards] <{unreached_object['class_name']}>({unreached_object['id']})")
            
            for next_room in self.next_rooms:
                if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
                    available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

        elif self.agent_node['class_name'] in ['robot arm', 'robot_arm']:
            if self.grabbed_objects is not None and self.on_surfaces is not None:
                available_plans.append(f"[puton] <{self.grabbed_objects['class_name']}>({self.grabbed_objects['id']}) on <{self.on_surfaces['class_name']}>({self.on_surfaces['id']})")
            
            for on_same_surface_object in self.on_same_surfaces:
                if self.grabbed_objects is None:
                    if 'CONTAINERS' in on_same_surface_object['properties']:
                        if 'OPEN' in on_same_surface_object['states']:
                            available_plans.append(f"[close] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
                        if 'CLOSED' in on_same_surface_object['states']:
                            available_plans.append(f"[open] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
                    if 'GRABABLE' in on_same_surface_object['properties']:
                        available_plans.append(f"[grab] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
                else:  # Has grabbed object
                    if 'CONTAINERS' in on_same_surface_object['properties'] and ('OPEN' in on_same_surface_object['states'] or "OPEN_FOREVER" in on_same_surface_object['states']):
                        available_plans.append(f"[putinto] <{self.grabbed_objects['class_name']}>({self.grabbed_objects['id']}) into <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
                    if 'SURFACES' in on_same_surface_object['properties']:
                        available_plans.append(f"[puton] <{self.grabbed_objects['class_name']}>({self.grabbed_objects['id']}) on <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"
        
        print(self.agent_node["class_name"], self.agent_node['id'])
        print(available_plans)
        
        return plans, len(available_plans), available_plans

    def parse_answer(self, available_actions, text):
        """Parse the LLM's answer to get the chosen action"""
        text = text.replace("_", " ")
        text = text.replace("takeoff from", "takeoff_from")
        text = text.replace("land on", "land_on")

        # Try direct action match
        for action in available_actions:
            if action in text:
                return action
        self.write_log_to_file('\nThe first action parsing failed!!!')

        # Try option letter match
        for i, action in enumerate(available_actions):
            option = chr(ord('A') + i)
            if any(opt in text for opt in [f"option {option}", f"{option}.", f"{option},", f"Option {option}", f"({option})"]):
                return action
        self.write_log_to_file('\nThe second action parsing failed!!!')

        print("WARNING! No available action parsed!!! Output plan NONE!\n")
        return None

    def write_log_to_file(self, log_message, file_name=None):
        """Write log message to file"""
        file_name = self.record_dir
        with open(file_name, 'a') as file:
            file.write(log_message + '\n')

    @backoff.on_exception(backoff.expo, Exception)
    def _generate(self, prompt, sampling_params):
        """Generate response from LLM with error handling and backoff"""
        usage = 0
        if self.source == 'openai':
            try:
                if self.chat:
                    prompt.insert(0, {"role": "system", "content": "You are a helper assistant."})
                    response = self.client.chat.completions.create(
                        model=self.lm_id, messages=prompt, **sampling_params
                    )
                    if self.debug:
                        with open(f"./chat_raw.json", 'a') as f:
                            f.write(json.dumps(response, indent=4))
                            f.write('\n')
                    generated_samples = [response.choices[i].message.content for i in range(sampling_params['n'])]
                    if 'gpt-4-0125-preview' in self.lm_id:
                        usage = response.usage.prompt_tokens * 0.01 / 1000 + response.usage.completion_tokens * 0.03 / 1000
                    elif 'gpt-3.5-turbo-1106' in self.lm_id:
                        usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
                else:
                    raise ValueError(f"{self.lm_id} not available!")
            except OpenAIError as e:
                print(e)
                raise e
        elif self.source == 'llm_module':
            try:
                if self.chat:
                    prompt.insert(0, {"role": "system", "content": "You are a helper assistant."})
                    response = self.client.respond_once_all_args(
                        messages=prompt, **sampling_params
                    )
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
                    raise ValueError(f"{self.lm_id} not available!")
            except Exception as e:
                print(e)
                raise e
        else:
            raise ValueError("invalid source")
        return generated_samples, usage