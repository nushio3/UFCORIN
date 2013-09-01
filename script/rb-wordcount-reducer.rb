#!/usr/bin/env ruby

accum = {}

while line = STDIN.gets
  xs = line.split("\t")
  accum[xs[0]] ||= 0
  accum[xs[0]] += xs[1].to_i
end

accum.each{|k,v|
  printf("%s\t%d\n", k, v) 
}
