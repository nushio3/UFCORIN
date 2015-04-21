module System.System where

import Control.Monad
import Control.Concurrent
import qualified Control.Exception as C
import Data.IORef
import System.Exit
import System.Process
import System.IO
import System.IO.Error

readSystem0 :: String -> IO String
readSystem0 cmd = readSystem cmd ""

readSystem
    :: String                   -- ^ shell command to run
    -> String                   -- ^ standard input
    -> IO String                -- ^ stdout + stderr
readSystem cmd input = do
    (Just inh, Just outh, _, pid) <-
        createProcess (shell cmd){ std_in  = CreatePipe,
                                       std_out = CreatePipe,
                                       std_err = Inherit }

    -- fork off a thread to start consuming the output
    output  <- hGetContents outh
    outMVar <- newEmptyMVar
    forkIO $ C.evaluate (length output) >> putMVar outMVar ()

    -- now write and flush any input
    when (not (null input)) $ do hPutStr inh input; hFlush inh
    hClose inh -- done with stdin

    -- wait on the output
    takeMVar outMVar
    hClose outh

    -- wait on the process
    ex <- waitForProcess pid

    case ex of
     ExitSuccess   -> return output
     ExitFailure r ->
      error ("readSystem: " ++ cmd ++
                                     " (exit " ++ show r ++ ")")
