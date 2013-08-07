module Main where

import Control.Monad
import Text.Printf

main :: IO ()
main = do
  putStrLn "\\documentclass{article}"
  putStrLn "\\usepackage{graphicx}"
  putStrLn "\\setlength{\\topmargin}{20mm}"
  putStrLn "\\addtolength{\\topmargin}{-1in}"
  putStrLn "\\setlength{\\oddsidemargin}{20mm}"
  putStrLn "\\addtolength{\\oddsidemargin}{-1in}"
  putStrLn "\\setlength{\\evensidemargin}{15mm}"
  putStrLn "\\addtolength{\\evensidemargin}{-1in}"
  putStrLn "\\setlength{\\textwidth}{170mm}"
  putStrLn "\\setlength{\\textheight}{254mm}"
  putStrLn "\\setlength{\\headsep}{0mm}"
  putStrLn "\\setlength{\\headheight}{0mm}"
  putStrLn "\\setlength{\\topskip}{0mm}"

  putStrLn "\\begin{document}"

  forM wavelets $ \waveletName ->  do
    printf "\\section{%s}\n" waveletName
    putStrLn "\\begin{center}"
    putStrLn "\\begin{tabular}{cc}"
    putStrLn "Quiet & Flare \\\\"

    let pairs = zip (take 4 targetFns) (drop 4 targetFns)
    forM pairs $ \(quietFn, flareFn) -> do
      printf "%s & %s \\\\\n" quietFn flareFn    
      let includePngFn fnstr = 
            printf "\\includegraphics[width=6cm]{dist/fwd-%s-%s.png}" waveletName fnstr
              :: String
      printf "%s & %s \\\\\n" 
        (includePngFn quietFn)
        (includePngFn flareFn)


    putStrLn "\\end{tabular}" 
    putStrLn "\\end{center}"   
  putStrLn "\\end{document}"

targetFns :: [FilePath]
targetFns =
  [ "20100609"
  , "20101013"
  , "20101109"
  , "20110511"
  , "20120306"
  , "20120309"
  , "20120509"
  , "20120701"
  ]


wavelets = 
  [ "N-bspl0-103"
  , "N-bspl0-309"
  , "N-bsplC-103"
  , "N-bsplC-309"
  , "N-daub0-20"
  , "N-daub0-4"
  , "N-daubC-20"
  , "N-daubC-4"
  , "N-haar0-2"
  , "N-haarC-2"
  , "S-bspl0-103"
  , "S-bspl0-309"
  , "S-bsplC-103"
  , "S-bsplC-309"
  , "S-daub0-20"
  , "S-daub0-4"
  , "S-daubC-20"
  , "S-daubC-4"
  , "S-haar0-2"
  , "S-haarC-2"
  ]