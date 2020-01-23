# zhetapi

## Goals ##

 - Enable less memory consuption by using pointers and freeing them when objects are destroyed
 - Enable memory saving features by creating a garbage collector class that will cache items into memory and reuse them later


## Benchmarks ##

 * January 22nd, 2020
   * Input: 0.123 to 18.5645 - (456 * 2 ^ 32 / 89)
   * Value Returned: -2.20057e+10
   * Memory Used (RAM): 1828 KB
   * Time Taken: 469.689 microseconds
 
