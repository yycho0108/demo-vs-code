#!/usr/bin/env python3

import xmlrpc.client

def main():
    proxy = xmlrpc.client.ServerProxy('http://localhost:8002/RPC2')

    while True:
        print(proxy.kpt())

if __name__ == '__main__':
    main()


