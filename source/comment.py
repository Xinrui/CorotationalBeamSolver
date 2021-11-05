    # @property
    # def local_stiffness_matrix(self):

    #     k_l = np.zeros((7, 7), dtype=float)

    #     k_l[0, 0] = self.__youngs_modulus * self.area

    #     k_l[1, 1], k_l[4, 4] = self.__shear_modulus * \
    #         self.polar_moment_of_inertia, self.__shear_modulus * self.polar_moment_of_inertia
    #     k_l[1, 4], k_l[4, 1] = -self.__shear_modulus * self.polar_moment_of_inertia, - \
    #         self.__shear_modulus * self.polar_moment_of_inertia

    #     k_l[2, 2], k_l[5, 5] = 4.0 * self.__youngs_modulus * \
    #         self.moment_of_inertia_z, 4.0 * self.__youngs_modulus * self.moment_of_inertia_z
    #     k_l[2, 5], k_l[5, 2] = 2.0 * self.__youngs_modulus * \
    #         self.moment_of_inertia_z, 2.0 * self.__youngs_modulus * self.moment_of_inertia_z

    #     k_l[3, 3], k_l[6, 6] = 4.0 * self.__youngs_modulus * \
    #         self.moment_of_inertia_y, 4.0 * self.__youngs_modulus * self.moment_of_inertia_y
    #     k_l[3, 6], k_l[6, 3] = 2.0 * self.__youngs_modulus * \
    #         self.moment_of_inertia_y, 2.0 * self.__youngs_modulus * self.moment_of_inertia_y

    #     k_l /= self.initial_length

    #     return k_l
    

    # @property
    # def local_force(self):
    #     # Assemble axial force and local end moments into a vector q_l.

    #     return self.local_stiffness_matrix @ self.local_displacement