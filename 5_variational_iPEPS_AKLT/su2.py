
import numpy as np

def get_su2_op(op, m, dbg = False):
  
    if op == "I":
        if dbg:
            print(">>>>> Constructing 1sO: Id <<<<<")
        return np.eye(m,m)
    
    elif op == "sz":
        if dbg:
            print(">>>>> Constructing 1sO: Sz <<<<<")
        res = np.zeros((m, m))
        for i in range(m):
            res[i,i] = -0.5 * (-(m - 1) + i*2)
        return res
    
    # The s^+ operator maps states with s^z = x to states with
    # s^z = x+1 . Therefore as a matrix it must act as follows
    # on vector of basis elements of spin S representation (in
    # this particular order) |S M>
    #
    #     |-S  >    C_+|-S+1>           0 1 0 0 ... 0
    # s^+ |-S+1>  = C_+|-S+2>  => S^+ = 0 0 1 0 ... 0 x C_+
    #      ...         ...              ...
    #     | S-1>    C_+| S  >           0    ...  0 1
    #     | S  >     0                  0    ...  0 0
    #
    # where C_+ = sqrt(S(S+1)-M(M+1))   
    elif op == "sp":
        if dbg:
            print(">>>>> Constructing 1sO: S^+ <<<<<")
        res = np.zeros((m, m))
        for i in range(m-1):
            res[i,i+1] = np.sqrt(0.5 * (m - 1) * (0.5 * (m - 1) + 1) - \
                     (-0.5 * (m - 1) + i) * \
                      (-0.5 * (m - 1) + i + 1))
        return res

    # The s^- operator maps states with s^z = x to states with
    # s^z = x-1 . Therefore as a matrix it must act as follows
    # on vector of basis elements of spin S representation (in
    # this particular order) |S M>
    #
    #     |-S  >     0                  0 0 0 0 ... 0
    # s^- |-S+1>  = C_-|-S  >  => S^- = 1 0 0 0 ... 0 x C_-
    #      ...         ...              ...
    #     | S-1>    C_-| S-2>           0   ... 1 0 0
    #     | S  >    C_-| S-1>           0   ... 0 1 0
    #
    # where C_- = sqrt(S(S+1)-M(M-1))
    elif op == "sm":
        if dbg:
            print(">>>>> Constructing 1sO: S^- <<<<<")
        res = np.zeros((m, m))
        for i in range(1,m):
            res[i, i - 1] = np.sqrt(0.5 * (m - 1) * (0.5 * (m - 1) + 1) - \
                     (-0.5 * (m - 1) + i) * \
                       (-0.5 * (m - 1) + i - 1))
        return res
    else:
        raise Exception("Unsupported operator requested: "+op)

def get_rot_op(m):
    res = np.zeros((m, m))
    for i in range(m):
        res[i,m-1-i] = np.power(-1,i)
    return res

# double SU2_getCG(int j1, int j2, int j, int m1, int m2, int m) {
#   // (!) Use Dynkin notation to pass desired irreps

#   double getCG = 0.0;
#   if (m == m1 + m2) {
#     double pref =
#       sqrt((j + 1.0) * Factorial((j + j1 - j2) / 2) *
#            Factorial((j - j1 + j2) / 2) * Factorial((j1 + j2 - j) / 2) /
#            Factorial((j1 + j2 + j) / 2 + 1)) *
#       sqrt(Factorial((j + m) / 2) * Factorial((j - m) / 2) *
#            Factorial((j1 - m1) / 2) * Factorial((j1 + m1) / 2) *
#            Factorial((j2 - m2) / 2) * Factorial((j2 + m2) / 2));
#     // write(*,'("<m1=",I2," m2=",I2,"|m=",I2,"> pref = ",1f10.5)') &
#     //      & m1, m2, m, pref
#     int min_k = min((j1 + j2) / 2, j2);
#     double sum_k = 0.0;
#     for (int k = 0; k <= min_k + 1; k++) {
#       if (((j1 + j2 - j) / 2 - k >= 0) && ((j1 - m1) / 2 - k >= 0) &&
#           ((j2 + m2) / 2 - k >= 0) && ((j - j2 + m1) / 2 + k >= 0) &&
#           ((j - j1 - m2) / 2 + k >= 0)) {
#         sum_k +=
#           pow(-1, k) /
#           (Factorial(k) * Factorial((j1 + j2 - j) / 2 - k) *
#            Factorial((j1 - m1) / 2 - k) * Factorial((j2 + m2) / 2 - k) *
#            Factorial((j - j2 + m1) / 2 + k) * Factorial((j - j1 - m2) / 2 + k));
#       }
#     }
#     getCG = pref * sum_k;
#   }
#   return getCG;
# }

# int Factorial(int x) {
#   if (x == 0) {
#     return 1;
#   } else if (x == 1) {
#     return 1;
#   }
#   return x * Factorial(x - 1);
# }
