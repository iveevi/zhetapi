# zhetapi

## Goals ##

 - Enable less memory consuption by using pointers and freeing them when objects are destroyed
 - Enable memory saving features by creating a garbage collector class that will cache items into memory and reuse them later

## Usage and Documentation ##

## Benchmarks ##

 * January 22nd, 2020 (Single Run):
   * Input: ```0.123 to 18.5645 - (456 * 2 ^ 32 / 89)```
   * Value Returned: ```-2.20057e+10```
   * Memory Used (RAM): ```1828 kilobytes```
   * Time Taken: ```469.689 microseconds```
 * Januray 23rd, 2020 (1000 Runs):
   * Input: ```0.123 to 18.5645 - (456 * 2 ^ 32 / 89)```
   * Value Returned: ```2.20057e+10```
   * Memory Used (RAM): ```4.87576e+06 kilobytes, 4875.76 kilobytes on average```
   * Time Taken: ```170.234 microseconds on average```