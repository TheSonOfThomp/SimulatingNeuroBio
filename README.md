# Simulating NeuroBiological Systems

Included are assignments for SYDE 556, and the Final Project.

For my final project, I implemented a crude version of auditory memory, and pitch recognition. I used the existing cochlea module created by [Trevor Bekolay](https://github.com/tbekolay) and linked it to a working memory system. By representing the input frequencies as n-dimensional vectors (where n is the granularity of the cochlea simulation) and comparing these vectors, I was able to implement a neural system which could determine whether a second pitch was higher or lower than the first.

From the abstract:

> The goal of this system is to take two consecutive audio signals as input, and determine whether the second note is higher or lower than the first. This system will consist of two main components—a cochlea and a working memory. The cochlea will convert the audio signal into a N-dimensional vector (where N is the number of hair cell groups modeled in the cochlea). The cochlea performs a frequency decomposition of the given sound into N frequency bins, and outputs a spike train with rates corresponding to the relative magnitudes of each frequency component.

> Without intensive training, or a neurological anomaly, there is generally no way to decode the root note (eg A4) of a pitch directly from the auditory nerve, typically called Perfect Pitch. With somewhat less training however, it is possible to develop Relative Pitch, which is the ability to determine the value of a given pitch given a reference pitch. (A version of this where the reference pitch is ingrained in long-term memory is typically indistinguishable form Perfect Pitch. Further, even an untrained listener (barring the condition "amusia") can determine whether a pitch is higher or lower than a prior pitch. This last example is the basis of the system.

> The N-dimensional vector representations of the input frequencies will be placed in a working-memory module. A comparison of the two stored vectors will be compared to output a final decision variable as the final response.
