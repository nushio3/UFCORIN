#!/bin/csh -f
#
# Simple script to fetch a JSOC recordset using calls to wget
#
# call with a two arguments containing a recordset query and your registered email address
# resulting data files are placed into the current directory
#
# see http://jsoc.stanford.edu/jsocwiki/AjaxConnect for further information
#
# e.g.
#   exportfile.csh 'hmi.M_45s[$]' XXX@YYY.ZZZ
# will get you the most recent SDO/HMI line-of-sight magnetogram if you replace the XXX@YYY.ZZZ with your
# registered email address (see below)

set noglob

# get desired recordset and encode for transmitting
# you can get the following url_escape script at http://jsoc.stanford.edu/ajax/

set ds=`/home/ubuntu/hub/UFCORIN/script/jsoc/url_escape.pl "$1"`

# You MUST now include a valid registered email address.  You can use a prior exportdata "notify" email address
# or just make one via http://jsoc2.stanford.edu/ajax/register_email.html
# Change the '$2' in the line below to your notify address if you want the script to not need a second argument.

set notify = `/home/ubuntu/hub/UFCORIN/script/jsoc/url_escape.pl "$2"`

# for FITS without full headers use fastest method use the next 2 lines:

set method=url_quick
set protocol=as-is

# if, as is most common, you want full FITS headers or
# if you want to specify the filename format for the returned data or if you expect multiple
# files you should use one of the next two methods.

# set method=url
# set protocol="FITS"

# to specify filename formats add the "filenamefmt" command to the cmd line below.
# you will need to url_escape the filenamefmt.

set ffmt = `/home/ubuntu/hub/UFCORIN/script/jsoc/url_escape.pl '{seriesname}.{T_REC:A}.{segment}'`

set op=exp_request

set cmd = "op=$op&ds=$ds&process=n=0|no_op&method=$method&format=txt&protocol=$protocol&filenamefmt=$ffmt&notify=$notify&requestor=none&sizeratio=1"

set JSOC="http://jsoc.stanford.edu"
set FETCH="$JSOC/cgi-bin/ajax/jsoc_fetch"

# Make a place to put the handshake information
set RESP = /tmp/jsoc_export.$$

wget -nv -O $RESP  $FETCH?"$cmd"

if ($method == "url") then
  set Status = `grep status $RESP | sed -e s/status/Status/ `
  set $Status
  if ($Status == 4) then
    echo "Export request failed, status=" $Status
    cat $RESP
    exit
  else if ($Status == 6) then
    echo "Problem with finding registered email address."
    cat $RESP
    exit
  endif

  set requestid = `grep requestid $RESP`
  if ($#requestid == 0) then
    echo no requestid found
    exit
  endif
  set $requestid

  # wait for processing to complete, do status query with sleeps and initial sleep
  set qry = "op=exp_status&requestid=$requestid&format=txt"
  while (1)
    sleep 3
    wget  -O $RESP $FETCH?"$qry"
    set Status = `grep status $RESP | sed -e s/status/Status/ `
    set $Status
    if ($Status == 0) break
    if ($Status == 6) then
        continue
    else if ($Status > 2) then
        echo Error in processing export request
        echo possibly bad recordset format or no data in SUMS
        cat $RESP
        exit
    endif
  end

else if ($method == "url_quick") then
  set Status = `grep status $RESP | sed -e s/status/Status/ `
  set $Status
  if ($Status == 4) then
    echo "Export request failed, status=" $Status
    cat $RESP
    exit
  else if ($Status == 6) then
    echo "Problem with finding registered email address."
    cat $RESP
    exit
  endif
endif

set dir = `grep dir $RESP`
set $dir
set count = `grep count $RESP`
set $count

# Now $RESP contains headers used above and table of count files.
# loop through files and fetch them.

while ($count > 0)
  set REC = `tail --lines=$count $RESP | head -1`
  set QUERY = $REC[1]
  set FILE = $REC[2]
  set URL = $JSOC$dir/$FILE
  set OFN = `/home/ubuntu/hub/UFCORIN/script/jsoc/url_escape.pl "$QUERY"`
  echo Fetching $QUERY as `basename $FILE`
  wget -nv $URL -O $OFN.fits
  @ count = $count - 1
  end

# remove working file.  Parhaps useful to save with the data.
# rm $RESP
mv $RESP ./


# The cgi-bin programs that you can use with wget or curl or such programs include
#   jsoc_fetch
#   jsoc_info
#   show_series
#   show_info
#
#   If you need to pass flags (e.g. -X) to these programs, use 'X=1' on the urlencoded command line.
#   To see the full set of options see the man pages in doxygen from the jsoc2 home page.
