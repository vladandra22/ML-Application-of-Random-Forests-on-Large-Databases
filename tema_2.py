import sys
import os
import tema_2_diabet
import tema_2_credit_risk

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['diabet', 'credit_risk']:
        print("Argumentul trebuie sa fie 'diabet' sau 'credit_risk'.")
        sys.exit(1)
    if sys.argv[1] == 'diabet':
        tema_2_diabet.solve()
    elif sys.argv[1] == 'credit_risk':
        tema_2_credit_risk.solve()
        
if __name__ == "__main__":
    main()