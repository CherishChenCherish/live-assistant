#!/usr/bin/env python3
"""Generate activation codes for customers.

Usage:
    python3 generate_codes.py customer@email.com           # 1 year
    python3 generate_codes.py customer@email.com 30        # 30 days
    python3 generate_codes.py --batch emails.txt           # bulk
"""

import sys
from license import generate_code


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_codes.py <email> [days]")
        print("       python3 generate_codes.py --batch <file.txt>")
        return

    if sys.argv[1] == "--batch":
        with open(sys.argv[2]) as f:
            for line in f:
                email = line.strip()
                if email:
                    code = generate_code(email, int(sys.argv[3]) if len(sys.argv) > 3 else 365)
                    print(f"{email} → {code}")
    else:
        email = sys.argv[1]
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
        code = generate_code(email, days)
        print(f"\nEmail: {email}")
        print(f"Days:  {days}")
        print(f"Code:  {code}")
        print(f"\nSend this code to the customer.")


if __name__ == "__main__":
    main()
