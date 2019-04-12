#!/usr/bin/python

import pexpect
import sys


if __name__ == '__main__':
    mypass = "9375"
    file_name = sys.argv[1]
    zip_name = sys.argv[2]
    shell_cmd = 'zip -e '+zip_name+file_name
    child = pexpect.spawn('/bin/bash', ['-c',shell_cmd])
    child.expect('(?i)password')
    child.sendline(mypass)
    child.expect('(?i)password')
    child.sendline(mypass)
    #print child.before   # Print the result of the ls command.
    child.interact()
