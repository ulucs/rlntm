# An Implementation of the RL Neural Turing Machine

## Action Space

|Operator|Corresponding Action|
|--------|--------------------|
|`>`|Moves on to the next character on the input string|
|`,`|Directly outputs the input character to output|
|`+`|Reads the input character into the external LIFO memory|
|`.`|Pushes out a character from external memory, or nothing if the external memory is empty|

## Implemented Tasks

|Task|Syntax|Output|Description|
|----|------|------|-----------|
|Copy|`c2345`|`2345`|Copies the input number directly to the output string|
|Reverse|`r124nnn`|`421`|Reverses the input number. Needs the end-of-line character `n` to be provided as our NTM takes one time step for each input character|
|Skip|`s123`|`23`|Copies the input number while omitting the first character|

## Usage

This implementation uses a feed-forward NTM in order to force the machine to use its external memory with the reverse task. After loading/training a model, the `runtm(model,inputstring)` function can be called to utilize the machine. The `runtm` function returns an output string interpreted by the model. If you want to see the actions taken by the NTM agent, you can use the `ntmactions(model,inputstring)` function instead.

## Example Trained Models

* `traineddata/shortntm.jld` contains the NTM model trained with reinforcre algorithm
 * contains 196 parameters
 * this model is able to do the copy and reverse tasks with short lengths(2) in most inputs
* `traineddata/perfect.jld` contains the NTM model trained directly
 * contains 196 parameters
 * this model is able to do the copy, skip and reverse tasks with any length (tested up to 100 length string in copy task)