# Facebook Message Analysis

Analysis done on my downloaded Facebook Messenger data (I mainly use messenger for talking to other people)

## Key Statistics
![Key Statistics](/graphs/text_table.png)

## Messages Sent Over Time (Binned by day)
![Messages sent over time ](/graphs/messages_over_time.png)

## Cumulative Messages Sent Over Time
![Cumulative Messages sent over time ](/graphs/cumulative_messages_over_time.png)

## Most Common Words
![Most Common Words](/graphs/common_words.png)

## Messages Sent by Day
![Messages Sent by Day](/graphs/messages_on_day.png)

## Messages Sent by Hour
![Messages Sent by Hour](/graphs/messages_on_hour.png)

## Messages Sent by Person
![Messages Sent by person](/graphs/messages_sent_person.png)

## Markov Chain Text Generator

Initially Markov chain using this implementiation:

		from collections import defaultdict, Counter
		import random
		import sys

		STATE_LEN = 4

		data = sys.stdin.read()
		model = defaultdict(Counter)

		for i in range(len(data) - STATE_LEN):
				state = data[i:i + STATE_LEN]
				next = data[i + STATE_LEN]
				model[state][next] += 1

		state = random.choice(list(model))
		out = list(state)
		for i in range(400):
				out.extend(random.choices(list(model[state]), model[state].values()))
				state = state[1:] + out[-1]
		print(''.join(out))
    
But later switched to Markovify python library.

Some examples of generated Markov text chains
![Markov text chains](/graphs/markov_gen.PNG)

