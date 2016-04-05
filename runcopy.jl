include("lstm.jl")

(data, testdata) = initcopy()
f = compile(:ntm;lsize=11,insize=11)
for n=1:50
	traintm(f,data)

	toterr = 0
	softer = 0
	for i=1:68
		yg = data[2][1]
		yt = forw(f,data[1][1])
		toterr += zeroone(yg,yt)
		softer += softloss(yg,yt)
	end
	reset!(f)
	softer = softer/68
	toterr = toterr/68

	toterrs = 0
	softers = 0
	for i=1:68
		yg = testdata[2][1]
		yt = forw(f,testdata[1][1])
		toterrs += zeroone(yg,yt)
		softers += softloss(yg,yt)
	end
	reset!(f)
	softers = softers/68
	toterrs = toterrs/68
	println("Epoch: ",n,"\nCurrent training accuracy: ",(1-toterr)*100,"%; softloss: ",softer,"\nCurrent test accuracy: ",(1-toterrs)*100,"%; softloss: ",softers)

	## if toterrs == 0
	## 	break
	## end
end