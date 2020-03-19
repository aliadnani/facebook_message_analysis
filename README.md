# Facebook Message Analysis

Analysis done on my downloaded Facebook Messenger data



## Key Statistics

### Using GROUPBY & COUNT functions

| Metric                                  | Value       |
|-----------------------------------------|-------------|
| Total Number of Messages Sent           | 145406      |
| Number of Photos Sent                   | 3179        |
| Number of Videos Sent                   | 141         |
| Total Number of Words Sent              | 2579381     |
| Total Number of Characters Sent         | 545485      |
| Average Number of Messages Sent per Day | 98.64721845 |
| Average Word Count per Message          | 3.751461425 |
| Average Character Count per Message     | 18.20966615 |

**Key Takeaways:**

- I text considerably more than the average person in my demographic (Every day people, aged 18–24, send **and receive** about 128 texts -> 64 sent per day); I am on 98.6
- **However,** my texting style is inconcise (Average Word Count per Message: 18.209),as often breaking up longer texts into shorter messages; eg (Hi! <send>, are you hungry? <send>, I'm going to cook <send>)
- Average of 1 photo sent per 50 text messages, interesting find: I thought this would be more.
	


### Messages Sent Over Time (Binned by day)
### From Mar 3, 2013 - Mar 19, 2020
  
I didn't have too many texting friends until high school where I began texting alot more. But texting in high school was mainly done over WhatsApp. I started using Messenger after starting university. 
  
![Messages sent over time ](/images/date_msgs.svg)

### Cumulative Messages Sent Over Time
### From Mar 3, 2013 - Mar 19, 2020
![Cumulative Messages sent over time ](/images/cumu_msgs.svg)
  

  
## Most Common Words
My day-to-day vocabulary is generally kept simple and casual.




| word | count |     |      |      |      |
|------|-------|-----|------|------|------|
| i    | 17451 | a   | 7888 | yeah | 4497 |
| you  | 11572 | it  | 7422 | its  | 4412 |
| like | 10337 | but | 6774 | that | 4303 |
| dude | 10333 | is  | 6382 | do   | 4294 |
| to   | 9456  | so  | 5180 | in   | 4137 |
| the  | 9429  | im  | 4552 |      |      |
| and  | 8158  | u   | 4507 |      |      |
  
![Most Common Words](/images/common_words.svg)
  
  
### Extras: Longest Words & Longest Message
The longest string I have sent is 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaa...' with a length of 461 'a's to spam my friend. Another notable mentions are 'hahahahahahahaha...' with a length of 312. Some Website links I have sent have also ended being long as well (+50 chars).

The longest valid word I have sent in which the word is a member of (https://github.com/dwyl/english-words) is a tie between 4 words that have 16 chars: 'entrepreneurship', 'responsibilities', 'semiprofessional', and 'electromagnetism'. It turns out I do not use that many big words in day to day texting. But then again only 1.708% of the words in the english words list have a length greater than 16, so I don't think I'm too dar off.

The longest message was an English essay assignment I sent over to a friend for their review, the length was around 6000 characters.


## Messages Sent by Day
| day_of_week | count   |
|-------------|---------|
| Monday      | 23858   |
| Tuesday     | 22548   |
| Wednesday   | 22214   |
| Thursday    | 23381   |
| Friday      | 18374   |
| Saturday    | 16822   |
| Sunday      | 18209   |
  
![Messages Sent by Day](/images/messages_on_day.svg)




## Messages Sent by Hour
  
I text alot throughout the day with peak texting time at 10pm and min time at 5am where I'm probably asleep

| hour_of_day | count | hour_of_day | count | hour_of_day | count | hour_of_day | count |
|-------------|-------|-------------|-------|-------------|-------|-------------|-------|
| 00          | 7444  | 06          | 1057  | 12          | 7922  | 18          | 10869 |
| 01          | 3608  | 07          | 1126  | 13          | 7511  | 19          | 12420 |
| 02          | 1178  | 08          | 1843  | 14          | 7285  | 20          | 11588 |
| 03          | 417   | 09          | 3160  | 15          | 6788  | 21          | 11334 |
| 04          | 318   | 10          | 4704  | 16          | 6612  | 22          | 12779 |
| 05          | 146   | 11          | 5448  | 17          | 9060  | 23          | 10789 |
  
![Messages Sent by Hour](/images/hour_msgs.svg)

## Messages Sent by Person
(Names Hidden)
![Messages Sent by person](/images/person2.svg)

## Markov Chain Text Generator

The original scope of this project was to generate a predictive model from my language style and how I text for generating text; Similar to the word prediction on your phone keyboard.

This implementation of a simple predictive model uses markov chains.

Some good resources:
https://en.wikipedia.org/wiki/Markov_chain
https://medium.com/analytics-vidhya/making-a-text-generator-using-markov-chains-e17a67225d10

Initially implemented Markov chains using this implementiation:

		from collections import defaultdict, Counter
		import random
		import sys

		STATE_LEN = 2

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
|   | 0                                                             |
|---|---------------------------------------------------------------|
| 0 | dont worry about money                                        |
| 1 | but its too close to when theyll get mad                      |
| 2 | i hope they get bangs                                         |
| 3 | it makes me angry                                             |
| 4 | but i think ill have noone to talk to her                     |
| 5 | and played football in class                                  |
| 6 | im trash at good grades this sem                              |
| 7 | so hes gotta be respectful                                    |
| 8 | cant you just write only potatoes as ur first paint job       |
| 9 | fillet is the universe disappear what would she take him back |

## LSTM Text Generation

There is also code in this project to train a LSTM model to generate text. Its mainly untested as I haven't gotten around to training a model to a high number of epochs.

The weights model that is supplied is trained for 1 epoch and is generally low quality usually generating strings like: 'ssddd sddff'

I intend to train this model further when I get tensorflow set up on my graphics-card-equipped desktop PC.

