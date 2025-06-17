proc highlighting { colorId representation id selection } {
   puts "highlighting $id"
   mol representation $representation
   mol material "Diffuse" 
    mol color $colorId
   mol selection $selection
   mol addrep $id
}

set id [mol new P95862_2_vina_out.pdb type pdb]
mol delrep top $id
highlighting Name "Lines" $id "protein"
highlighting Name "Licorice" $id "not protein and not resname STP"
highlighting Element "NewCartoon" $id "protein"
highlighting "ColorID 7" "VdW 0.4" $id "protein and occupancy>0.95"
set id [mol new P95862_2_vina_pockets.pqr type pqr]
                        mol selection "all" 
                         mol material "Glass3" 
                         mol delrep top $id 
                         mol representation "QuickSurf 0.3" 
                         mol color ResId $id 
                         mol addrep $id 
highlighting Index "Points 1" $id "resname STP"
proc highlighting { colorId representation id selection } {
   puts "highlighting $id"
   mol representation $representation
   mol material "Diffuse" 
    mol color $colorId
   mol selection $selection
   mol addrep $id
}

set id [mol new P95862_2_vina_out.pdb type pdb]
mol delrep top $id
highlighting Name "Lines" $id "protein"
highlighting Name "Licorice" $id "not protein and not resname STP"
highlighting Element "NewCartoon" $id "protein"
highlighting "ColorID 7" "VdW 0.4" $id "protein and occupancy>0.95"

# Load the PQR file, disable auto bond guessing
set pid [mol new P95862_2_vina_pockets.pqr type pqr autobonds off]

# Pocket surface rendering
mol selection "all"
mol material "Glass3"
mol delrep $pid 0
mol representation "QuickSurf 0.3"
mol color ResId
mol addrep $pid

# Highlight STP residues
highlighting Index "Points 1" $pid "resname STP"

# Use OpenGL instead of GLSL for safer rendering
display rendermode OpenGL

