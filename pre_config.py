##this is the number of distinct labels
##the extrator will only read 
##NUM_LABELS-1 most frequent classes
##and assign "OtherType" to everything else
NUM_LABELS=1000

# truncate the longer ones, pad the shorter ones
# the defaults are chosen so that about 1% of the
# names is truncated, everything else is padded with 
# PAD_CHARACTER
fixlens = {
	"funcName": 46, #default 46
	"argName": 23, #default 23
	"allArgs": 320, #default 320
	"assigns": 300, #default 300 assuming a two character separator
	"returnStatements": 250, #default 250
	"docstring": 100, #default 100
}

PAD_CHARACTER='\0'
#concatenate different information
SEPARATOR = "$^"
#concat info from one list, such as allArgs
SEPARATOR_MINOR = ";;"

#default 40
#maximum:
#   ret_data: sum (fixlens.items()[1]) - fixlens["docstring"] - fixlens["argName"] + 4 * len(SEPARATOR) #default 924
#   arg_data: sum (fixlens.items()[1]) - fixlens["docstring"] + 5 * len (SEPARATOR) #default 949
SLIDING_WINDOW_SIZE=40

STEP_SIZE=3