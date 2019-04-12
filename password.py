#!/usr/bin/python
import pexpect
import test
import sys

if __name__ == '__main__':
    user = sys.argv[1]
    zip_name = sys.argv[2]
    mypass = test.pass_in(user)
    shell_cmd = 'unzip '+zip_name
    child = pexpect.spawn('/bin/bash', ['-c',shell_cmd])
    child.expect('(?i)password')
    child.sendline(mypass)
    #print child.before   # Print the result of the ls command.
    child.interact()
    pass
