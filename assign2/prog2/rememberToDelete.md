# Prog2 Base Ideas


### Use Async Communication

Don't think barriers are necessary


#### Rank 0
- create an array with file results

- Each file result contains Filename,  array of determinants, order of the matrix, number of matrix

- Continously send chunks of data

- Send matrix, order of the matrix, number and fileindex


- When finished send final finish message

- Wait to receive all data from workers,
wait for final worker message to end

- If all workers send 'finished' message, show results and finish process


#### Other ranks

- Receive chunks of data

- Calculate the determinant for each chunk

- Send results

- If receives a finish message, end process

