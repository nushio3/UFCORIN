#!/usr/bin/env ruby

str = STDIN.read
str.split.each{|w|
  printf("%s\t1\n",w)
}
