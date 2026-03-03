# TODO this should go to tests
# %% some plots to verify these functions
# a coordinate system
# basis = np.identity(2)
# origin = np.array([4, 4])
# cs1 = CoordinateSystem(basis=basis, origin=origin)

# # another coordinate system, translated
# origin_t = np.array([2, 2])
# basis_t = np.identity(2)
# basis_t[0,0] *= 2

# cs2 = CoordinateSystem(basis=basis_t, origin=origin_t)
# cs = LinkedCoordinateSystems(dict(original=cs1, translated=cs2))

# axes = cs.plot(color_by="axis", scale=1)

# cs.transform(np.array([0, 0]), "original", "translated") # -2, -2
# cs.transform(np.array([-1, -1]), "translated", "original")


# # %%
# # another coordinate system, rotated
# # origin = np.ones(2) + 2
# # basis_r = np.array([[[1,-1],[]]])

# # fail on rotating non-uniform coordinate axes!


# # rotatian: translate to origin, apply rotation matrix, translate back
# def rotation_matrix_2d(theta):
#     rotation_matrix = np.array(
#         [
#             [np.cos(theta), -np.sin(theta)],
#             [
#                 np.sin(theta),
#                 np.cos(theta),
#             ],
#         ]
#     )
#     return rotation_matrix


# basis_r = (basis_t.T @ rotation_matrix_2d(np.pi / 4)).T  # transpose because we have column vectors

# cs3 = CoordinateSystem(basis=basis_r, origin=np.ones(2) + 3, name="rotated")

# cs = LinkedCoordinateSystems([cs1, cs2, cs3])
# axes = cs.plot(color_by="axis")

# cs.transform(np.array([0, 1]), "translated", "original")

# axes.set_xlabel("x")
# axes.set_ylabel("y")

# %%
