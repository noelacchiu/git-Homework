def plota_f(x0, xf, n):
  xpoints = []
  ypoints = []
  for i in range(0,n):
      xpoints.append(x0+(xf-x0)/n *i)
      ypoints.append(np.sin(x0+(xf-x0)/n *i))
  plt.plot(xpoints, ypoints)
  plt.show()